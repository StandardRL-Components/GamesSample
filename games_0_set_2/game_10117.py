import gymnasium as gym
import os
import pygame
import pygame.gfxdraw
import math
import numpy as np
from gymnasium.spaces import MultiDiscrete, Box
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    GameEnv: A fast-paced cosmic cruiser racing and combat game.
    The player pilots a neon cruiser, racing against rivals across a procedurally
    generated starfield littered with obstacles. The goal is to reach the finish
    line first, using weapons and gadgets to gain an edge.
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    game_description = (
        "A fast-paced cosmic cruiser racing and combat game. Pilot a neon cruiser, "
        "race against rivals, and use weapons and gadgets to reach the finish line first."
    )
    user_guide = (
        "Controls: Use ↑↓←→ arrow keys to accelerate, brake, and turn. "
        "Press space to fire your weapon and shift to activate your boost."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Game & World Constants ---
        self.W, self.H = 640, 400
        self.WORLD_WIDTH = 6000
        self.FINISH_LINE_X = self.WORLD_WIDTH - 500
        self.MAX_STEPS = 2000
        self.NUM_ENEMIES = 3
        self.NUM_OBSTACLES = 40
        self.NUM_STARS = 200

        # --- Visuals & Colors ---
        self.COLOR_BG = (10, 5, 25)
        self.COLOR_PLAYER = (0, 255, 150)
        self.COLOR_PLAYER_GLOW = (0, 255, 150, 50)
        self.ENEMY_COLORS = [(255, 80, 80), (80, 150, 255), (255, 255, 80)]
        self.COLOR_PROJECTILE = (255, 150, 0)
        self.COLOR_GADGET_BOOST = (0, 220, 255)
        self.COLOR_OBSTACLE = (100, 80, 120)
        self.COLOR_FINISH_LINE = (255, 255, 255)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.COLOR_HEALTH_FG = (255, 60, 60)
        self.COLOR_HEALTH_BG = (50, 50, 50)

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.W, self.H))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 16, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_over_message = ""
        self.player = {}
        self.enemies = []
        self.projectiles = []
        self.particles = []
        self.obstacles = []
        self.stars = []
        self.camera_pos = np.array([0.0, 0.0])
        self.prev_dist_to_finish = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_over_message = ""

        self._generate_world()

        # --- Initialize Player ---
        self.player = self._create_cruiser(is_player=True, rank=0)

        # --- Initialize Enemies ---
        self.enemies = [self._create_cruiser(is_player=False, rank=i + 1) for i in range(self.NUM_ENEMIES)]

        self.projectiles = []
        self.particles = []

        self.prev_dist_to_finish = self._get_dist_to_finish(self.player)

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0.0
        terminated = False

        if not self.game_over:
            # --- Unpack Actions ---
            movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

            # --- Update Game Logic ---
            self._update_cruiser(self.player, movement, space_held, shift_held)
            reward += self._update_enemies_ai()
            reward += self._update_projectiles()
            self._update_particles()
            self._handle_collisions()

            # --- Continuous Rewards & Step Counter ---
            self.steps += 1
            reward -= 0.01  # Time penalty

            current_dist = self._get_dist_to_finish(self.player)
            reward += (self.prev_dist_to_finish - current_dist) * 0.1
            self.prev_dist_to_finish = current_dist

        # --- Check for Termination ---
        terminated, term_reward = self._check_termination()
        reward += term_reward
        if terminated and not self.game_over:
            self.game_over = True
            self.score += term_reward

        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False, # truncated
            self._get_info()
        )

    def _get_observation(self):
        self._update_camera()

        self.screen.fill(self.COLOR_BG)

        self._render_background()
        self._render_world()
        self._render_projectiles_and_particles()

        # Render enemies first, then player on top
        for enemy in self.enemies:
            if enemy['health'] > 0:
                self._draw_cruiser(self.screen, enemy)
        if self.player['health'] > 0:
            self._draw_cruiser(self.screen, self.player)

        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_health": self.player['health'],
            "distance_to_finish": self._get_dist_to_finish(self.player),
        }

    # --- ENTITY CREATION & WORLD GENERATION ---

    def _create_cruiser(self, is_player, rank):
        base_health = 100 if is_player else 80
        difficulty_health_mod = 1.0 + (self.steps // 200) * 0.01

        return {
            "is_player": is_player,
            "pos": np.array([100.0, self.H / 2 + (rank - self.NUM_ENEMIES / 2) * 60]),
            "vel": np.array([0.0, 0.0]),
            "angle": 0.0,
            "health": base_health * (1 if is_player else difficulty_health_mod),
            "max_health": base_health * (1 if is_player else difficulty_health_mod),
            "color": self.COLOR_PLAYER if is_player else self.ENEMY_COLORS[rank - 1],
            "size": 15 if is_player else 14,
            "weapon_cooldown": 0,
            "gadget_cooldown": 0,
            "gadget_active_timer": 0,
            "rank": rank
        }

    def _generate_world(self):
        # Generate stars with parallax effect
        self.stars = []
        for _ in range(self.NUM_STARS):
            self.stars.append({
                'pos': np.array([self.np_random.uniform(0, self.WORLD_WIDTH), self.np_random.uniform(0, self.H)]),
                'depth': self.np_random.uniform(0.1, 0.6),  # Slower moving stars are 'further'
                'size': self.np_random.uniform(1, 2.5)
            })

        # Generate obstacles, avoiding the start area
        self.obstacles = []
        for _ in range(self.NUM_OBSTACLES):
            pos = np.array([
                self.np_random.uniform(400, self.FINISH_LINE_X - 100),
                self.np_random.uniform(50, self.H - 50)
            ])
            radius = self.np_random.uniform(20, 50)
            self.obstacles.append({'pos': pos, 'radius': radius})

    # --- GAME LOGIC UPDATES ---

    def _update_cruiser(self, cruiser, movement, space_held, shift_held):
        if cruiser['health'] <= 0: return

        # --- Gadget Effects ---
        base_accel = 0.25
        base_turn_speed = 0.05
        base_max_speed = 4.0
        drag = 0.96

        if cruiser['gadget_active_timer'] > 0:
            base_max_speed *= 1.8
            base_accel *= 1.5
            drag = 0.98  # Less drag during boost
            cruiser['gadget_active_timer'] -= 1

        # --- Handle Input ---
        if movement == 1:  # Accelerate
            accel_vec = np.array([math.cos(cruiser['angle']), math.sin(cruiser['angle'])]) * base_accel
            cruiser['vel'] += accel_vec
        elif movement == 2:  # Brake/Reverse
            accel_vec = np.array([math.cos(cruiser['angle']), math.sin(cruiser['angle'])]) * -base_accel * 0.7
            cruiser['vel'] += accel_vec
        if movement == 3:  # Turn Left
            cruiser['angle'] -= base_turn_speed
        if movement == 4:  # Turn Right
            cruiser['angle'] += base_turn_speed

        # --- Firing ---
        if cruiser['weapon_cooldown'] > 0: cruiser['weapon_cooldown'] -= 1
        if space_held and cruiser['weapon_cooldown'] == 0:
            proj_start_pos = cruiser['pos'] + np.array(
                [math.cos(cruiser['angle']), math.sin(cruiser['angle'])]) * cruiser['size']
            self.projectiles.append({
                'pos': proj_start_pos,
                'vel': np.array([math.cos(cruiser['angle']), math.sin(cruiser['angle'])]) * 8.0 + cruiser['vel'],
                'owner': cruiser,
                'lifespan': 60
            })
            cruiser['weapon_cooldown'] = 10 if cruiser['is_player'] else 25

        # --- Gadget ---
        if cruiser['gadget_cooldown'] > 0: cruiser['gadget_cooldown'] -= 1
        if shift_held and cruiser['gadget_cooldown'] == 0 and cruiser['gadget_active_timer'] == 0:
            cruiser['gadget_active_timer'] = 90  # 3 seconds at 30fps
            cruiser['gadget_cooldown'] = 300  # 10 second cooldown
            if cruiser['is_player']: self.score += 10  # Reward for using gadget

            # Add boost particles
            for _ in range(30):
                angle = cruiser['angle'] + math.pi + self.np_random.uniform(-0.5, 0.5)
                speed = self.np_random.uniform(2, 5)
                vel = np.array([math.cos(angle), math.sin(angle)]) * speed
                self.particles.append({
                    'pos': cruiser['pos'].copy(), 'vel': vel, 'radius': self.np_random.uniform(2, 4),
                    'color': self.COLOR_GADGET_BOOST, 'lifespan': 20
                })

        # --- Physics Update ---
        cruiser['vel'] *= drag
        speed = np.linalg.norm(cruiser['vel'])
        if speed > base_max_speed:
            cruiser['vel'] = (cruiser['vel'] / speed) * base_max_speed

        cruiser['pos'] += cruiser['vel']

        # Boundary checks
        cruiser['pos'][0] = np.clip(cruiser['pos'][0], 0, self.WORLD_WIDTH)
        cruiser['pos'][1] = np.clip(cruiser['pos'][1], 0, self.H)

    def _update_enemies_ai(self):
        reward = 0
        difficulty_speed_mod = 1.0 + (self.steps // 100) * 0.05

        for enemy in self.enemies:
            if enemy['health'] <= 0: continue

            # Simple AI: move towards finish line with some wobble and player avoidance
            target_y = self.H / 2 + math.sin(self.steps / 50.0 + enemy['rank']) * 100
            target_dir = np.array([self.FINISH_LINE_X, target_y]) - enemy['pos']
            target_angle = math.atan2(target_dir[1], target_dir[0])

            angle_diff = (target_angle - enemy['angle'] + math.pi) % (2 * math.pi) - math.pi

            movement = 1  # Always accelerate
            if angle_diff < -0.1:
                movement = 3  # Turn left
            elif angle_diff > 0.1:
                movement = 4  # Turn right

            # Firing logic
            player_dir = self.player['pos'] - enemy['pos']
            dist_to_player = np.linalg.norm(player_dir)
            should_fire = False
            if dist_to_player < 300:
                angle_to_player = math.atan2(player_dir[1], player_dir[0])
                if abs((angle_to_player - enemy['angle'] + math.pi) % (2 * math.pi) - math.pi) < 0.2:
                    should_fire = True

            # Apply difficulty scaling to speed
            original_vel = enemy['vel'].copy()
            self._update_cruiser(enemy, movement, should_fire, False)
            enemy['vel'] = original_vel + (enemy['vel'] - original_vel) * difficulty_speed_mod
        return reward

    def _update_projectiles(self):
        reward = 0
        for proj in self.projectiles[:]:
            proj['pos'] += proj['vel']
            proj['lifespan'] -= 1
            if proj['lifespan'] <= 0 or not (0 < proj['pos'][0] < self.WORLD_WIDTH and 0 < proj['pos'][1] < self.H):
                self.projectiles.remove(proj)
                continue

            # --- Projectile Collision ---
            targets = self.enemies if proj['owner']['is_player'] else [self.player]
            for target in targets:
                if target['health'] <= 0: continue
                dist = np.linalg.norm(proj['pos'] - target['pos'])
                if dist < target['size']:
                    damage = 10
                    target['health'] -= damage
                    if proj['owner']['is_player']:
                        reward += 1  # Damage reward

                    if target['health'] <= 0:
                        target['health'] = 0
                        if proj['owner']['is_player']:
                            reward += 5  # Destruction reward
                        self._add_particles(target['pos'], target['color'], 50, 4, 30)
                    else:
                        self._add_particles(proj['pos'], self.COLOR_PROJECTILE, 10, 2, 15)

                    if proj in self.projectiles:
                        self.projectiles.remove(proj)
                    break
        return reward

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            p['radius'] *= 0.95
            if p['lifespan'] <= 0:
                self.particles.remove(p)

    def _handle_collisions(self):
        all_cruisers = [self.player] + self.enemies
        for cruiser in all_cruisers:
            if cruiser['health'] <= 0: continue
            # Obstacle collisions
            for obs in self.obstacles:
                dist = np.linalg.norm(cruiser['pos'] - obs['pos'])
                if dist < cruiser['size'] + obs['radius']:
                    overlap = (cruiser['size'] + obs['radius']) - dist
                    normal = (cruiser['pos'] - obs['pos']) / dist
                    cruiser['pos'] += normal * overlap
                    cruiser['vel'] = cruiser['vel'] - 2 * np.dot(cruiser['vel'], normal) * normal * 0.5  # bounce
                    cruiser['vel'] *= 0.8  # speed penalty

    def _check_termination(self):
        # 1. Player destroyed
        if self.player['health'] <= 0:
            self.game_over_message = "CRUISER DESTROYED"
            return True, -50

        # 2. Player wins
        if self.player['pos'][0] >= self.FINISH_LINE_X:
            all_cruisers = [self.player] + self.enemies
            sorted_cruisers = sorted(all_cruisers, key=lambda c: c['pos'][0], reverse=True)
            player_rank = sorted_cruisers.index(self.player) + 1
            if player_rank == 1:
                self.game_over_message = "VICTORY!"
                return True, 100
            else:
                self.game_over_message = f"FINISHED #{player_rank}"
                return True, 50

        # 3. Max steps reached
        if self.steps >= self.MAX_STEPS:
            self.game_over_message = "TIME LIMIT REACHED"
            return True, -10

        # 4. Fell too far behind
        max_x = max(c['pos'][0] for c in [self.player] + self.enemies if c['health'] > 0)
        if self.player['pos'][0] < max_x - self.W * 1.5:
            self.game_over_message = "LEFT BEHIND"
            return True, -25

        return False, 0

    # --- RENDERING ---

    def _update_camera(self):
        # Smoothly follow the player
        target_cam_x = self.player['pos'][0] - self.W / 3
        target_cam_y = self.player['pos'][1] - self.H / 2
        self.camera_pos[0] = 0.9 * self.camera_pos[0] + 0.1 * target_cam_x
        self.camera_pos[1] = 0.9 * self.camera_pos[1] + 0.1 * target_cam_y

        # Clamp camera
        self.camera_pos[0] = max(0, self.camera_pos[0])
        self.camera_pos[1] = max(0, self.camera_pos[1])

    def _render_background(self):
        for star in self.stars:
            # Parallax scrolling
            screen_x = (star['pos'][0] - self.camera_pos[0] * star['depth']) % self.W
            screen_y = (star['pos'][1] - self.camera_pos[1] * star['depth']) % self.H
            # Dim stars based on depth
            color_val = int(200 * star['depth'])
            pygame.draw.circle(self.screen, (color_val, color_val, color_val), (int(screen_x), int(screen_y)),
                               int(star['size']))

    def _render_world(self):
        # Render obstacles
        for obs in self.obstacles:
            screen_pos = obs['pos'] - self.camera_pos
            if -obs['radius'] < screen_pos[0] < self.W + obs['radius'] and -obs['radius'] < screen_pos[1] < self.H + \
                    obs['radius']:
                pygame.gfxdraw.filled_circle(self.screen, int(screen_pos[0]), int(screen_pos[1]), int(obs['radius']),
                                             self.COLOR_OBSTACLE)

        # Render finish line
        finish_screen_x = self.FINISH_LINE_X - self.camera_pos[0]
        if 0 < finish_screen_x < self.W:
            for y in range(0, self.H, 20):
                color = self.COLOR_FINISH_LINE if (y // 20) % 2 == 0 else self.COLOR_BG
                pygame.draw.rect(self.screen, color, (finish_screen_x, y, 10, 20))

    def _render_projectiles_and_particles(self):
        # Render particles
        for p in self.particles:
            screen_pos = p['pos'] - self.camera_pos
            alpha_color = (*p['color'], int(255 * (p['lifespan'] / 20)))
            pygame.gfxdraw.filled_circle(self.screen, int(screen_pos[0]), int(screen_pos[1]),
                                         max(0, int(p['radius'])), alpha_color)

        # Render projectiles
        for proj in self.projectiles:
            screen_pos = proj['pos'] - self.camera_pos
            pygame.draw.circle(self.screen, self.COLOR_PROJECTILE, (int(screen_pos[0]), int(screen_pos[1])), 3)

    def _draw_cruiser(self, surface, cruiser):
        screen_pos = cruiser['pos'] - self.camera_pos
        size = cruiser['size']
        angle = cruiser['angle']

        # Points for the triangular ship body
        p1 = (screen_pos[0] + math.cos(angle) * size, screen_pos[1] + math.sin(angle) * size)
        p2 = (screen_pos[0] + math.cos(angle + 2.2) * size, screen_pos[1] + math.sin(angle + 2.2) * size)
        p3 = (screen_pos[0] + math.cos(angle - 2.2) * size, screen_pos[1] + math.sin(angle - 2.2) * size)
        points = [p1, p2, p3]

        # Glow effect
        glow_color = (*cruiser['color'][:3], 50)
        pygame.gfxdraw.filled_polygon(surface, [(int(p[0]), int(p[1])) for p in points], glow_color)

        # Main body
        pygame.gfxdraw.aapolygon(surface, [(int(p[0]), int(p[1])) for p in points], cruiser['color'])
        pygame.gfxdraw.filled_polygon(surface, [(int(p[0]), int(p[1])) for p in points], cruiser['color'])

        # Engine trail/glow
        if np.linalg.norm(cruiser['vel']) > 0.5:
            num_flame = 3 if cruiser['gadget_active_timer'] > 0 else 1
            for i in range(num_flame):
                flame_angle = angle + math.pi + self.np_random.uniform(-0.2, 0.2)
                flame_len = size * self.np_random.uniform(0.8, 1.5 + (2.0 if cruiser['gadget_active_timer'] > 0 else 0))
                flame_p1 = (p2[0] + p3[0]) / 2, (p2[1] + p3[1]) / 2
                flame_p2 = (flame_p1[0] + math.cos(flame_angle) * flame_len, flame_p1[1] + math.sin(flame_angle) * flame_len)
                flame_color = self.COLOR_GADGET_BOOST if cruiser['gadget_active_timer'] > 0 else (255, 100, 0)
                pygame.draw.line(surface, flame_color, flame_p1, flame_p2, 2)

        # Health bar
        if not cruiser['is_player']:
            health_pct = cruiser['health'] / cruiser['max_health']
            bar_w, bar_h = 30, 4
            bar_x, bar_y = screen_pos[0] - bar_w / 2, screen_pos[1] - size - 10
            pygame.draw.rect(surface, self.COLOR_HEALTH_BG, (bar_x, bar_y, bar_w, bar_h))
            pygame.draw.rect(surface, self.COLOR_HEALTH_FG, (bar_x, bar_y, bar_w * health_pct, bar_h))

    def _render_ui(self):
        # Score and Steps
        self._draw_text(f"SCORE: {int(self.score)}", (10, 10))
        self._draw_text(f"STEPS: {self.steps}/{self.MAX_STEPS}", (self.W - 150, 10))

        # Player Health Bar
        health_pct = self.player['health'] / self.player['max_health']
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, (10, self.H - 30, 200, 20))
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, (10, self.H - 30, max(0, 200 * health_pct), 20))
        self._draw_text(f"HEALTH", (15, self.H - 28))

        # Gadget Cooldown Indicator
        gadget_text = "BOOST [SHIFT]"
        color = self.COLOR_GADGET_BOOST
        if self.player['gadget_cooldown'] > 0:
            pct = self.player['gadget_cooldown'] / 300
            gadget_text = f"COOLDOWN {int(pct * 100)}%"
            color = (80, 80, 90)
        elif self.player['gadget_active_timer'] > 0:
            gadget_text = "BOOST ACTIVE"

        self._draw_text(gadget_text, (self.W - 150, self.H - 28), color=color)

        # Game Over Message
        if self.game_over:
            s = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
            s.fill((0, 0, 0, 180))
            self.screen.blit(s, (0, 0))
            text_surf = self.font_large.render(self.game_over_message, True, self.COLOR_UI_TEXT)
            text_rect = text_surf.get_rect(center=(self.W / 2, self.H / 2))
            self.screen.blit(text_surf, text_rect)

    def _draw_text(self, text, pos, color=None, font=None):
        font = font or self.font_small
        color = color or self.COLOR_UI_TEXT
        text_surface = font.render(text, True, color)
        self.screen.blit(text_surface, pos)

    # --- UTILITY & HELPERS ---

    def _add_particles(self, pos, color, count, speed_max, lifespan):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, speed_max)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            self.particles.append({
                'pos': pos.copy(), 'vel': vel, 'radius': self.np_random.uniform(1, 4),
                'color': color, 'lifespan': self.np_random.integers(lifespan // 2, lifespan)
            })

    def _get_dist_to_finish(self, cruiser):
        return max(0, self.FINISH_LINE_X - cruiser['pos'][0])

    def close(self):
        pygame.quit()


# --- Example Usage ---
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")

    # --- Manual Play ---
    # Controls:
    #   Arrows: Move
    #   Space:  Fire
    #   Shift:  Boost
    #   R:      Reset
    #   Q:      Quit

    obs, info = env.reset()
    running = True

    # We need a Pygame window to display the rendered array
    pygame.display.set_caption("Cosmic Cruiser Clash")
    screen = pygame.display.set_mode((env.W, env.H))

    while running:
        # --- Action Mapping for Human Player ---
        keys = pygame.key.get_pressed()
        movement = 0  # None
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4

        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement, space_held, shift_held]

        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                if event.key == pygame.K_r:
                    obs, info = env.reset()

        # --- Step Environment ---
        obs, reward, terminated, truncated, info = env.step(action)

        # --- Render to screen ---
        # The observation is already the rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(env.metadata["render_fps"])

    env.close()