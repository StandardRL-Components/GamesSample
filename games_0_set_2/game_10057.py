import gymnasium as gym
import os
import pygame
import numpy as np
import pygame.gfxdraw
import math
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from gymnasium.spaces import MultiDiscrete, Box


class GameEnv(gym.Env):
    """
    A Gymnasium environment where a morphing creature collects fragments,
    evades enemies, and unlocks new forms. The goal is to survive for 60 seconds
    and collect 20 fragments.
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

    game_description = (
        "Control a morphing creature to collect fragments and evade enemies. "
        "Survive for 60 seconds and collect 20 fragments to win."
    )
    user_guide = (
        "Use arrow keys (↑↓←→) to move. Press space to dash and shift to change form."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 3600  # 60 seconds at 60 FPS
    VICTORY_TIME_SECONDS = 60
    VICTORY_FRAGMENTS = 20
    NUM_FRAGMENTS = 25
    NUM_ENEMIES = 5
    NUM_OBSTACLES = 8

    # --- Colors ---
    COLOR_BG = (25, 20, 40)
    COLOR_GRID = (35, 30, 50)
    COLOR_OBSTACLE = (80, 80, 90)
    COLOR_ENEMY = (255, 50, 50)
    COLOR_FRAGMENT_PALETTE = [(0, 255, 255), (255, 0, 255), (255, 255, 0)]
    COLOR_TEXT = (220, 220, 240)
    COLOR_TEXT_SHADOW = (10, 10, 20)

    # --- Player Forms ---
    FORMS = [
        {
            "name": "Walker",
            "color": (50, 150, 255),
            "vertices": 4,
            "speed": 300.0,
            "friction": 0.92,
            "dash_power": 400.0,
            "dash_cooldown": 45,
        },
        {
            "name": "Sprinter",
            "color": (255, 200, 50),
            "vertices": 3,
            "speed": 400.0,
            "friction": 0.95,
            "dash_power": 300.0,
            "dash_cooldown": 60,
        },
        {
            "name": "Dasher",
            "color": (150, 50, 255),
            "vertices": 8,
            "speed": 250.0,
            "friction": 0.90,
            "dash_power": 600.0,
            "dash_cooldown": 30,
        },
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 16)
        
        self.render_mode = render_mode
        self.window = None
        self.dt = 1 / self.metadata["render_fps"]

        # Initialize all state variables to prevent uninitialized attribute errors
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_timer = 0.0
        self.collected_fragments = 0
        self.player_pos = np.zeros(2, dtype=np.float32)
        self.player_vel = np.zeros(2, dtype=np.float32)
        self.player_size = 12
        self.current_form_idx = 0
        self.unlocked_forms = 1
        self.is_morphing = False
        self.morph_progress = 0.0
        self.target_form_idx = 0
        self.dash_cooldown_timer = 0
        self.prev_shift_held = False
        self.obstacles = []
        self.fragments = []
        self.enemies = []
        self.particles = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_timer = 0.0
        self.collected_fragments = 0

        self.player_pos = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2], dtype=np.float32)
        self.player_vel = np.zeros(2, dtype=np.float32)

        self.current_form_idx = 0
        self.unlocked_forms = 1
        self.is_morphing = False
        self.morph_progress = 0.0
        self.target_form_idx = 0
        self.dash_cooldown_timer = 0
        self.prev_shift_held = False

        self._generate_level()

        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0.0
        self.steps += 1
        self.game_timer += self.dt

        self._handle_input(movement, space_held, shift_held)
        self._update_player()
        self._update_enemies()
        self._update_particles()
        
        # Check for fragment collection
        collected_this_step = self._handle_fragment_collisions()
        if collected_this_step > 0:
            reward += 0.1 * collected_this_step
            self.score += 10 * collected_this_step
            self.collected_fragments += collected_this_step
            
            # Check for form unlock
            new_unlocked_forms = min(len(self.FORMS), 1 + self.collected_fragments // 20)
            if new_unlocked_forms > self.unlocked_forms:
                self.unlocked_forms = new_unlocked_forms
                reward += 5.0
                self.score += 500

        # Check for enemy collision
        if self._handle_enemy_collisions():
            self.game_over = True
            reward -= 50.0

        # Check for termination conditions
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        
        # Victory condition
        if self.game_timer >= self.VICTORY_TIME_SECONDS and self.collected_fragments >= self.VICTORY_FRAGMENTS:
            reward += 50.0
            terminated = True

        self.prev_shift_held = shift_held

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _generate_level(self):
        self.obstacles = []
        for _ in range(self.NUM_OBSTACLES):
            w = self.np_random.integers(40, 150)
            h = self.np_random.integers(40, 150)
            x = self.np_random.integers(0, self.SCREEN_WIDTH - w)
            y = self.np_random.integers(0, self.SCREEN_HEIGHT - h)
            self.obstacles.append(pygame.Rect(x, y, w, h))

        self.fragments = []
        while len(self.fragments) < self.NUM_FRAGMENTS:
            size = 8
            x = self.np_random.integers(size, self.SCREEN_WIDTH - size)
            y = self.np_random.integers(size, self.SCREEN_HEIGHT - size)
            frag_rect = pygame.Rect(x - size // 2, y - size // 2, size, size)
            
            if not any(obs.colliderect(frag_rect) for obs in self.obstacles):
                if frag_rect.colliderect(pygame.Rect(self.player_pos[0]-16, self.player_pos[1]-16, 32, 32)):
                    continue
                self.fragments.append(frag_rect)

        self.enemies = []
        while len(self.enemies) < self.NUM_ENEMIES:
            size = 10
            path_type = self.np_random.choice(["h", "v"])
            start_pos = np.array([
                self.np_random.integers(size, self.SCREEN_WIDTH - size),
                self.np_random.integers(size, self.SCREEN_HEIGHT - size)
            ], dtype=np.float32)
            
            enemy_rect = pygame.Rect(start_pos[0]-size, start_pos[1]-size, size*2, size*2)
            if any(obs.colliderect(enemy_rect) for obs in self.obstacles):
                continue

            self.enemies.append({
                "pos": start_pos,
                "base_pos": start_pos.copy(),
                "size": size,
                "path_type": path_type,
                "amplitude": self.np_random.uniform(50, 150),
                "frequency": self.np_random.uniform(0.5, 1.5),
                "phase": self.np_random.uniform(0, 2 * math.pi)
            })

    def _handle_input(self, movement, space_held, shift_held):
        form = self.FORMS[self.current_form_idx]
        accel = np.zeros(2, dtype=np.float32)
        
        if movement == 1: accel[1] -= 1  # Up
        if movement == 2: accel[1] += 1  # Down
        if movement == 3: accel[0] -= 1  # Left
        if movement == 4: accel[0] += 1  # Right
        
        if np.linalg.norm(accel) > 0:
            accel = accel / np.linalg.norm(accel) * form["speed"]
        self.player_vel += accel * self.dt

        # Dash action
        if space_held and self.dash_cooldown_timer <= 0:
            if np.linalg.norm(self.player_vel) > 0:
                direction = self.player_vel / np.linalg.norm(self.player_vel)
            else:
                direction = np.array([1, 0], dtype=np.float32)
            self.player_vel += direction * form["dash_power"]
            self.dash_cooldown_timer = form["dash_cooldown"]
            self._create_particle_burst(self.player_pos, 20, form["color"], 2.0, 5.0)

        # Form change action (on key press)
        if shift_held and not self.prev_shift_held and not self.is_morphing and self.unlocked_forms > 1:
            self.is_morphing = True
            self.morph_progress = 0.0
            self.target_form_idx = (self.current_form_idx + 1) % self.unlocked_forms

    def _update_player(self):
        form = self.FORMS[self.current_form_idx]
        
        if self.dash_cooldown_timer > 0:
            self.dash_cooldown_timer -= 1

        self.player_vel *= (form["friction"] ** (self.dt * 60))

        next_pos = self.player_pos + self.player_vel * self.dt
        
        player_rect = pygame.Rect(next_pos[0] - self.player_size, next_pos[1] - self.player_size, self.player_size*2, self.player_size*2)
        
        for obs in self.obstacles:
            if obs.colliderect(player_rect):
                player_rect.centerx = self.player_pos[0]
                if obs.colliderect(player_rect):
                    self.player_vel[1] *= -0.5
                    next_pos[1] = self.player_pos[1]
                
                player_rect.centerx = next_pos[0]
                player_rect.centery = self.player_pos[1]
                if obs.colliderect(player_rect):
                    self.player_vel[0] *= -0.5
                    next_pos[0] = self.player_pos[0]

        self.player_pos = next_pos

        self.player_pos[0] %= self.SCREEN_WIDTH
        self.player_pos[1] %= self.SCREEN_HEIGHT
        
        if np.linalg.norm(self.player_vel) > 50:
            self._create_particles(self.player_pos, 1, form["color"], 0.5, 1.0, 20)
            
        if self.is_morphing:
            self.morph_progress += self.dt * 4.0
            if self.morph_progress >= 1.0:
                self.morph_progress = 1.0
                self.is_morphing = False
                self.current_form_idx = self.target_form_idx

    def _update_enemies(self):
        for enemy in self.enemies:
            t = self.game_timer * enemy["frequency"] + enemy["phase"]
            offset = enemy["amplitude"] * math.sin(t)
            if enemy["path_type"] == 'h':
                enemy["pos"][0] = enemy["base_pos"][0] + offset
            else:
                enemy["pos"][1] = enemy["base_pos"][1] + offset
            
            enemy["pos"][0] = np.clip(enemy["pos"][0], enemy["size"], self.SCREEN_WIDTH - enemy["size"])
            enemy["pos"][1] = np.clip(enemy["pos"][1], enemy["size"], self.SCREEN_HEIGHT - enemy["size"])

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel'] * self.dt
            p['life'] -= self.dt
            p['size'] = max(0, p['start_size'] * (p['life'] / p['max_life']))

    def _handle_fragment_collisions(self):
        player_rect = pygame.Rect(0, 0, self.player_size * 1.5, self.player_size * 1.5)
        player_rect.center = tuple(self.player_pos)
        
        collected_indices = player_rect.collidelistall(self.fragments)
        if not collected_indices:
            return 0
        
        for i in sorted(collected_indices, reverse=True):
            frag_rect = self.fragments.pop(i)
            self._create_particle_burst(np.array(frag_rect.center), 30, self.COLOR_FRAGMENT_PALETTE[i % 3], 1.0, 4.0, 40)
        
        return len(collected_indices)

    def _handle_enemy_collisions(self):
        player_circle_pos = self.player_pos
        player_radius = self.player_size
        for enemy in self.enemies:
            dist = np.linalg.norm(player_circle_pos - enemy["pos"])
            if dist < player_radius + enemy["size"]:
                self._create_particle_burst(self.player_pos, 50, self.FORMS[self.current_form_idx]['color'], 2.0, 6.0, 60)
                self._create_particle_burst(enemy["pos"], 50, self.COLOR_ENEMY, 2.0, 6.0, 60)
                return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_obstacles()
        self._render_fragments()
        self._render_enemies()
        self._render_particles()
        self._render_player()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "collected_fragments": self.collected_fragments,
            "game_timer": self.game_timer,
            "unlocked_forms": self.unlocked_forms
        }

    def _render_background(self):
        for x in range(0, self.SCREEN_WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT), 1)
        for y in range(0, self.SCREEN_HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y), 1)
            
    def _render_obstacles(self):
        for obs in self.obstacles:
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, obs)

    def _render_fragments(self):
        for i, frag in enumerate(self.fragments):
            color = self.COLOR_FRAGMENT_PALETTE[i % 3]
            center = frag.center
            
            pygame.gfxdraw.filled_circle(self.screen, center[0], center[1], 10, (*color, 30))
            pygame.gfxdraw.filled_circle(self.screen, center[0], center[1], 7, (*color, 60))
            
            pygame.draw.rect(self.screen, color, frag)

    def _render_enemies(self):
        for enemy in self.enemies:
            pos_int = (int(enemy["pos"][0]), int(enemy["pos"][1]))
            pulse = 1 + 0.2 * math.sin(self.game_timer * 5)
            size = int(enemy["size"] * pulse)
            
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], int(size * 2.0), (*self.COLOR_ENEMY, 30))
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], int(size * 1.5), (*self.COLOR_ENEMY, 60))

            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], size, self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], size, self.COLOR_ENEMY)

    def _render_particles(self):
        for p in self.particles:
            if p['size'] > 1:
                pos_int = (int(p['pos'][0]), int(p['pos'][1]))
                alpha = min(255, int(255 * (p['life'] / p['max_life'])))
                color = (*p['color'], alpha)
                pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], int(p['size']), color)

    def _render_player(self):
        if self.game_over:
            return
            
        pos_int = (int(self.player_pos[0]), int(self.player_pos[1]))
        
        current_form = self.FORMS[self.current_form_idx]
        if self.is_morphing:
            target_form = self.FORMS[self.target_form_idx]
            p = self.morph_progress
            
            p = p * p * (3 - 2 * p)
            
            v_current = current_form['vertices']
            v_target = target_form['vertices']
            
            color = tuple(int(c1 * (1 - p) + c2 * p) for c1, c2 in zip(current_form['color'], target_form['color']))
            
            points1 = self._get_polygon_points(v_current, self.player_size, 0)
            points2 = self._get_polygon_points(v_target, self.player_size, 0)
            
            if v_current < v_target:
                points1 = [points1[i % v_current] for i in range(v_target)]
            elif v_target < v_current:
                points2 = [points2[i % v_target] for i in range(v_current)]
            
            final_points = []
            for i in range(len(points1)):
                px = points1[i][0] * (1 - p) + points2[i][0] * p + pos_int[0]
                py = points1[i][1] * (1 - p) + points2[i][1] * p + pos_int[1]
                final_points.append((int(px), int(py)))

        else:
            color = current_form['color']
            rotation = self.game_timer * 2
            final_points = [(p[0] + pos_int[0], p[1] + pos_int[1]) for p in self._get_polygon_points(current_form['vertices'], self.player_size, rotation)]

        pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], int(self.player_size * 2.5), (*color, 30))
        pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], int(self.player_size * 1.8), (*color, 60))
        
        pygame.gfxdraw.aapolygon(self.screen, final_points, color)
        pygame.gfxdraw.filled_polygon(self.screen, final_points, color)

    def _render_ui(self):
        frag_text = f"FRAGMENTS: {self.collected_fragments}/{self.VICTORY_FRAGMENTS}"
        self._draw_text(frag_text, (10, 10), self.font_main)
        
        time_left = max(0, self.VICTORY_TIME_SECONDS - self.game_timer)
        timer_text = f"TIME: {time_left:.1f}"
        self._draw_text(timer_text, (self.SCREEN_WIDTH - 150, 10), self.font_main, align="left")
        
        form = self.FORMS[self.current_form_idx]
        form_name = f"FORM: {form['name']}"
        self._draw_text(form_name, (10, self.SCREEN_HEIGHT - 30), self.font_small)
        
        cooldown_bar_width = 100
        cooldown_fill = cooldown_bar_width * (1 - min(1, self.dash_cooldown_timer / form["dash_cooldown"])) if form["dash_cooldown"] > 0 else cooldown_bar_width
        pygame.draw.rect(self.screen, self.COLOR_TEXT_SHADOW, (150, self.SCREEN_HEIGHT - 25, cooldown_bar_width, 10))
        pygame.draw.rect(self.screen, form['color'], (150, self.SCREEN_HEIGHT - 25, cooldown_fill, 10))

    def _get_polygon_points(self, vertices, radius, angle_offset):
        return [
            (
                math.cos(angle_offset + 2 * math.pi * i / vertices) * radius,
                math.sin(angle_offset + 2 * math.pi * i / vertices) * radius,
            )
            for i in range(vertices)
        ]

    def _create_particles(self, pos, count, color, min_speed, max_speed, max_life_frames):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(min_speed, max_speed)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed * 60
            max_life = self.np_random.uniform(0.5, 1.0) * max_life_frames / 60.0
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'color': color,
                'life': max_life,
                'max_life': max_life,
                'start_size': self.np_random.uniform(2, 5),
                'size': 0
            })

    def _create_particle_burst(self, pos, count, color, min_speed, max_speed, max_life_frames=30):
        self._create_particles(pos, count, color, min_speed, max_speed, max_life_frames)
        
    def _draw_text(self, text, pos, font, color=COLOR_TEXT, shadow_color=COLOR_TEXT_SHADOW, align="left"):
        text_surf = font.render(text, True, color)
        shadow_surf = font.render(text, True, shadow_color)
        text_rect = text_surf.get_rect()
        if align == "left":
            text_rect.topleft = pos
        elif align == "right":
            text_rect.topright = pos
        
        self.screen.blit(shadow_surf, (text_rect.x + 2, text_rect.y + 2))
        self.screen.blit(text_surf, text_rect)

    def close(self):
        if self.window is not None:
            pygame.display.quit()
        pygame.quit()

if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    
    terminated = False
    total_reward = 0.0
    
    pygame.init()
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Morph Creature")
    clock = pygame.time.Clock()
    
    while not terminated:
        movement = 0
        space_held = 0
        shift_held = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1

        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(GameEnv.metadata["render_fps"])

        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")

    env.close()