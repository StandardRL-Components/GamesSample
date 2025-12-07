import gymnasium as gym
import os
import pygame
import numpy as np
import pygame.gfxdraw
import math
import random
from gymnasium.spaces import MultiDiscrete
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A Gymnasium environment for a game set on a cell membrane.
    The player controls an amoeba-like entity to repair breaches and fight pathogens.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Control an amoeba-like entity to defend a cell membrane. "
        "Repair breaches and fight off invading pathogens using collected resources."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move. Press space to deploy an antibody "
        "against pathogens and shift to deploy a signal to repair breaches."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    MAX_STEPS = 1500

    # Colors
    COLOR_BG = (15, 10, 40)
    COLOR_BG_CELLS = [(25, 20, 50, 50), (30, 15, 60, 50)]
    COLOR_PLAYER = (0, 255, 255)
    COLOR_PLAYER_GLOW = (0, 150, 150)
    COLOR_PATHOGEN = (180, 0, 255)
    COLOR_PATHOGEN_SPIKE = (140, 0, 200)
    COLOR_BREACH = (255, 50, 50)
    COLOR_BREACH_PARTICLE = (200, 40, 40)
    COLOR_ANTIBODY_PICKUP = (50, 150, 255)
    COLOR_SIGNAL_PICKUP = (255, 255, 0)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_UI_BAR_BG = (50, 50, 50)
    COLOR_UI_BAR_FILL = (0, 200, 100)
    COLOR_UI_BAR_EMPTY = (200, 0, 50)

    # Player settings
    PLAYER_SPEED = 5
    PLAYER_RADIUS = 12

    # Game mechanics settings
    INITIAL_BREACHES = 2
    INITIAL_RESOURCES = 5
    MAX_RESOURCES_ON_SCREEN = 10
    PATHOGEN_SPAWN_INTERVAL = 120  # steps
    BREACH_SPAWN_INTERVAL = 450 # steps
    PATHOGEN_BASE_SPEED = 1.0
    PATHOGEN_RADIUS = 10
    BREACH_RADIUS = 25
    BREACH_DAMAGE_PER_STEP = 0.02
    PATHOGEN_CONTACT_DAMAGE = 10.0
    DEPLOY_COOLDOWN = 15 # steps
    DEPLOY_EFFECT_DURATION = 20 # steps
    DEPLOY_EFFECT_MAX_RADIUS = 60

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        self.render_mode = render_mode

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 20, bold=True)

        # Pre-generate background for performance
        self.background_surface = self._create_background()

        # Initialize state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = np.array([0.0, 0.0])
        self.membrane_integrity = 100.0
        self.resources = {}
        self.breaches = []
        self.pathogens = []
        self.resource_pickups = []
        self.deployed_effects = []
        self.particles = []
        self.previous_space_held = False
        self.previous_shift_held = False
        self.cooldowns = {}
        self.timers = {}
        self.total_reward = 0.0


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.total_reward = 0.0
        self.game_over = False

        self.player_pos = np.array([self.SCREEN_WIDTH / 2.0, self.SCREEN_HEIGHT / 2.0])
        self.membrane_integrity = 100.0

        self.resources = {'antibody': 3, 'signal': 3}
        self.cooldowns = {'antibody': 0, 'signal': 0}
        self.timers = {
            'pathogen_spawn': self.PATHOGEN_SPAWN_INTERVAL,
            'breach_spawn': self.BREACH_SPAWN_INTERVAL
        }

        self.previous_space_held = False
        self.previous_shift_held = False

        self.breaches = []
        for _ in range(self.INITIAL_BREACHES):
            self._spawn_breach()

        self.pathogens = []
        self._spawn_pathogen()

        self.resource_pickups = []
        for _ in range(self.INITIAL_RESOURCES):
            self._spawn_resource()
            
        self.deployed_effects = []
        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0

        # --- 1. Handle Input and Player Actions ---
        self._handle_input(action)

        # --- 2. Update Game State ---
        self._update_timers()
        self._update_pathogens()
        self._update_effects_and_particles()

        # --- 3. Handle Collisions and Game Logic ---
        reward += self._handle_collisions()
        reward += self._apply_continuous_damage()

        self.total_reward += reward

        # --- 4. Check for Termination ---
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS

        if terminated or truncated:
            if self.membrane_integrity <= 0:
                reward -= 100 # -100 for losing
            elif not self.breaches and terminated: # Win condition only if not truncated
                reward += 100 # +100 for winning
                self.score += 1 # A "win" is worth 1 point
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Movement
        move_vec = np.array([0.0, 0.0])
        if movement == 1: move_vec[1] -= 1 # Up
        elif movement == 2: move_vec[1] += 1 # Down
        elif movement == 3: move_vec[0] -= 1 # Left
        elif movement == 4: move_vec[0] += 1 # Right
        
        if np.linalg.norm(move_vec) > 0:
            move_vec = move_vec / np.linalg.norm(move_vec)
        self.player_pos += move_vec * self.PLAYER_SPEED
        self._wrap_around(self.player_pos)

        # Cooldowns
        for key in self.cooldowns:
            self.cooldowns[key] = max(0, self.cooldowns[key] - 1)

        # Deploy Antibody (Space)
        space_pressed = space_held and not self.previous_space_held
        if space_pressed and self.resources['antibody'] > 0 and self.cooldowns['antibody'] == 0:
            self.resources['antibody'] -= 1
            self.cooldowns['antibody'] = self.DEPLOY_COOLDOWN
            self.deployed_effects.append({
                'pos': self.player_pos.copy(), 'type': 'antibody', 'age': 0, 'max_age': self.DEPLOY_EFFECT_DURATION
            })
            self._create_particles(self.player_pos, self.COLOR_ANTIBODY_PICKUP, 10, 2.0)

        # Deploy Signal (Shift)
        shift_pressed = shift_held and not self.previous_shift_held
        if shift_pressed and self.resources['signal'] > 0 and self.cooldowns['signal'] == 0:
            self.resources['signal'] -= 1
            self.cooldowns['signal'] = self.DEPLOY_COOLDOWN
            self.deployed_effects.append({
                'pos': self.player_pos.copy(), 'type': 'signal', 'age': 0, 'max_age': self.DEPLOY_EFFECT_DURATION
            })
            self._create_particles(self.player_pos, self.COLOR_SIGNAL_PICKUP, 10, 2.0)

        self.previous_space_held = space_held
        self.previous_shift_held = shift_held

    def _update_timers(self):
        # Pathogen Spawning
        self.timers['pathogen_spawn'] -= 1
        if self.timers['pathogen_spawn'] <= 0:
            self._spawn_pathogen()
            self.timers['pathogen_spawn'] = self.PATHOGEN_SPAWN_INTERVAL

        # Breach Spawning
        self.timers['breach_spawn'] -= 1
        if self.timers['breach_spawn'] <= 0 and len(self.breaches) < 5:
            self._spawn_breach()
            self.timers['breach_spawn'] = self.BREACH_SPAWN_INTERVAL
        
        # Resource Spawning
        if len(self.resource_pickups) < self.MAX_RESOURCES_ON_SCREEN and self.np_random.random() < 0.01:
            self._spawn_resource()

    def _update_pathogens(self):
        speed_multiplier = 1.0 + (self.steps / 200.0) * 0.1
        for p in self.pathogens:
            p['pos'] += p['vel'] * self.PATHOGEN_BASE_SPEED * speed_multiplier
            self._wrap_around(p['pos'])
            p['angle'] = (p['angle'] + p['rot_speed']) % 360

    def _update_effects_and_particles(self):
        self.deployed_effects = [e for e in self.deployed_effects if e['age'] < e['max_age']]
        for e in self.deployed_effects:
            e['age'] += 1

        self.particles = [p for p in self.particles if p['lifetime'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifetime'] -= 1

    def _handle_collisions(self):
        reward = 0
        
        # Player vs Resource Pickups
        for r_pickup in self.resource_pickups[:]:
            if self._check_collision(self.player_pos, self.PLAYER_RADIUS, r_pickup['pos'], 10):
                self.resource_pickups.remove(r_pickup)
                self.resources[r_pickup['type']] += 1
                reward += 0.1

        # Player vs Pathogens
        for p in self.pathogens:
            if self._check_collision(self.player_pos, self.PLAYER_RADIUS, p['pos'], self.PATHOGEN_RADIUS):
                self.membrane_integrity -= self.PATHOGEN_CONTACT_DAMAGE
                reward -= 1.0 # Larger penalty for direct contact
                self._create_particles(self.player_pos, self.COLOR_BREACH, 20, 3.0)
        
        # Deployed Effects vs Game Objects
        for effect in self.deployed_effects:
            effect_radius = (effect['age'] / effect['max_age']) * self.DEPLOY_EFFECT_MAX_RADIUS
            
            # Antibody vs Pathogens
            if effect['type'] == 'antibody':
                for p in self.pathogens[:]:
                    if self._check_collision(effect['pos'], effect_radius, p['pos'], self.PATHOGEN_RADIUS):
                        self.pathogens.remove(p)
                        reward += 1.0
                        self.score += 1
                        self._create_particles(p['pos'], self.COLOR_PATHOGEN, 30, 4.0)

            # Signal vs Breaches
            if effect['type'] == 'signal':
                for b in self.breaches[:]:
                    if self._check_collision(effect['pos'], effect_radius, b['pos'], self.BREACH_RADIUS):
                        self.breaches.remove(b)
                        reward += 5.0
                        self.score += 5
                        self._create_particles(b['pos'], self.COLOR_SIGNAL_PICKUP, 50, 2.0)
        
        return reward

    def _apply_continuous_damage(self):
        reward = 0
        if self.breaches:
            damage = len(self.breaches) * self.BREACH_DAMAGE_PER_STEP
            self.membrane_integrity -= damage
            reward -= 0.1 # Penalty for existing damage
        self.membrane_integrity = max(0, self.membrane_integrity)
        return reward

    def _check_termination(self):
        if self.membrane_integrity <= 0:
            return True
        if not self.breaches and self.steps > 1: # Win condition
            return True
        return False

    # --- Spawning Methods ---

    def _spawn_breach(self):
        pos = self._get_random_screen_pos(margin=50)
        points = []
        num_points = self.np_random.integers(7, 11)
        for i in range(num_points):
            angle = (i / num_points) * 2 * math.pi
            radius = self.BREACH_RADIUS * self.np_random.uniform(0.7, 1.3)
            points.append((math.cos(angle) * radius, math.sin(angle) * radius))
        self.breaches.append({'pos': pos, 'points': points})

    def _spawn_pathogen(self):
        edge = self.np_random.integers(4)
        if edge == 0: # top
            pos = np.array([self.np_random.uniform(0, self.SCREEN_WIDTH), -self.PATHOGEN_RADIUS])
        elif edge == 1: # bottom
            pos = np.array([self.np_random.uniform(0, self.SCREEN_WIDTH), self.SCREEN_HEIGHT + self.PATHOGEN_RADIUS])
        elif edge == 2: # left
            pos = np.array([-self.PATHOGEN_RADIUS, self.np_random.uniform(0, self.SCREEN_HEIGHT)])
        else: # right
            pos = np.array([self.SCREEN_WIDTH + self.PATHOGEN_RADIUS, self.np_random.uniform(0, self.SCREEN_HEIGHT)])
        
        target = self.player_pos + self.np_random.uniform(-50, 50, size=2)
        vel = target - pos
        vel = vel / np.linalg.norm(vel) if np.linalg.norm(vel) > 0 else np.array([1.0, 0.0])
        
        self.pathogens.append({
            'pos': pos, 'vel': vel, 'angle': 0, 'rot_speed': self.np_random.uniform(-5, 5)
        })

    def _spawn_resource(self):
        pos = self._get_random_screen_pos(margin=30)
        r_type = 'antibody' if self.np_random.random() < 0.5 else 'signal'
        self.resource_pickups.append({'pos': pos, 'type': r_type})

    # --- Helper Methods ---

    def _get_random_screen_pos(self, margin=0):
        return np.array([
            self.np_random.uniform(margin, self.SCREEN_WIDTH - margin),
            self.np_random.uniform(margin, self.SCREEN_HEIGHT - margin)
        ])
    
    def _wrap_around(self, pos_vec):
        pos_vec[0] %= self.SCREEN_WIDTH
        pos_vec[1] %= self.SCREEN_HEIGHT

    def _check_collision(self, pos1, r1, pos2, r2):
        dist_sq = np.sum((pos1 - pos2)**2)
        return dist_sq < (r1 + r2)**2

    def _create_particles(self, pos, color, count, max_speed):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(0.5, max_speed)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'lifetime': self.np_random.integers(15, 30),
                'color': color,
                'radius': self.np_random.uniform(1, 3)
            })
            
    # --- Rendering ---

    def _get_observation(self):
        self.screen.blit(self.background_surface, (0, 0))
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "integrity": self.membrane_integrity}

    def _create_background(self):
        bg = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        bg.fill(self.COLOR_BG)
        for _ in range(50):
            pos = (random.randint(0, self.SCREEN_WIDTH), random.randint(0, self.SCREEN_HEIGHT))
            radius = random.randint(50, 150)
            color = random.choice(self.COLOR_BG_CELLS)
            pygame.gfxdraw.filled_circle(bg, pos[0], pos[1], radius, color)
        return bg

    def _render_game(self):
        # Breaches
        for b in self.breaches:
            points = [(p[0] + b['pos'][0], p[1] + b['pos'][1]) for p in b['points']]
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_BREACH)
            if self.np_random.random() < 0.5:
                p_pos = b['pos'] + self.np_random.uniform(-self.BREACH_RADIUS, self.BREACH_RADIUS, size=2)
                self.particles.append({
                    'pos': p_pos, 'vel': self.np_random.uniform(-0.5, 0.5, size=2),
                    'lifetime': 15, 'color': self.COLOR_BREACH_PARTICLE, 'radius': 2
                })

        # Resource Pickups
        for r in self.resource_pickups:
            pos = (int(r['pos'][0]), int(r['pos'][1]))
            color = self.COLOR_ANTIBODY_PICKUP if r['type'] == 'antibody' else self.COLOR_SIGNAL_PICKUP
            self._draw_glowing_shape(pos, 8, color, is_triangle=(r['type'] == 'signal'))
            
        # Deployed Effects
        for e in self.deployed_effects:
            color = self.COLOR_ANTIBODY_PICKUP if e['type'] == 'antibody' else self.COLOR_SIGNAL_PICKUP
            radius = int((e['age'] / e['max_age']) * self.DEPLOY_EFFECT_MAX_RADIUS)
            alpha = int(100 * (1 - e['age'] / e['max_age']))
            self._draw_circle_alpha(self.screen, color + (alpha,), (int(e['pos'][0]), int(e['pos'][1])), radius)

        # Pathogens
        for p in self.pathogens:
            pos_int = (int(p['pos'][0]), int(p['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], self.PATHOGEN_RADIUS, self.COLOR_PATHOGEN)
            for i in range(5):
                angle = math.radians(p['angle'] + i * (360 / 5))
                p1 = (pos_int[0] + math.cos(angle) * self.PATHOGEN_RADIUS, pos_int[1] + math.sin(angle) * self.PATHOGEN_RADIUS)
                p2 = (pos_int[0] + math.cos(angle) * (self.PATHOGEN_RADIUS + 5), pos_int[1] + math.sin(angle) * (self.PATHOGEN_RADIUS + 5))
                pygame.draw.line(self.screen, self.COLOR_PATHOGEN_SPIKE, p1, p2, 2)

        # Particles
        for p in self.particles:
            alpha = p['lifetime'] / 30.0
            color = (*p['color'], int(alpha * 255))
            self._draw_circle_alpha(self.screen, color, (int(p['pos'][0]), int(p['pos'][1])), int(p['radius']))

        # Player
        player_pos_int = (int(self.player_pos[0]), int(self.player_pos[1]))
        self._draw_glowing_shape(player_pos_int, self.PLAYER_RADIUS, self.COLOR_PLAYER)

    def _render_ui(self):
        # Membrane Integrity Bar
        bar_x, bar_y, bar_w, bar_h = 10, 10, 200, 20
        integrity_ratio = self.membrane_integrity / 100.0
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR_BG, (bar_x, bar_y, bar_w, bar_h))
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR_EMPTY, (bar_x, bar_y, bar_w, bar_h))
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR_FILL, (bar_x, bar_y, int(bar_w * integrity_ratio), bar_h))
        
        integrity_text = self.font_small.render(f"INTEGRITY", True, self.COLOR_UI_TEXT)
        self.screen.blit(integrity_text, (bar_x + 5, bar_y + 2))

        # Resource Counts
        res_y = bar_y + bar_h + 10
        # Antibody
        pygame.gfxdraw.filled_circle(self.screen, bar_x + 10, res_y + 8, 8, self.COLOR_ANTIBODY_PICKUP)
        antibody_text = self.font_small.render(f"x {self.resources['antibody']}", True, self.COLOR_UI_TEXT)
        self.screen.blit(antibody_text, (bar_x + 25, res_y))
        # Signal
        self._draw_glowing_shape((bar_x + 80, res_y + 8), 8, self.COLOR_SIGNAL_PICKUP, is_triangle=True, glow=False)
        signal_text = self.font_small.render(f"x {self.resources['signal']}", True, self.COLOR_UI_TEXT)
        self.screen.blit(signal_text, (bar_x + 95, res_y))
        
        # Score
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10, 10))

    def _draw_glowing_shape(self, pos, radius, color, is_triangle=False, glow=True):
        if glow:
            glow_radius = int(radius * 1.8)
            glow_color = (*color, 60)
            self._draw_circle_alpha(self.screen, glow_color, pos, glow_radius)

        if is_triangle:
            points = [
                (pos[0], pos[1] - radius),
                (pos[0] - radius * 0.866, pos[1] + radius * 0.5),
                (pos[0] + radius * 0.866, pos[1] + radius * 0.5)
            ]
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
        else:
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, color)

    def _draw_circle_alpha(self, surface, color, center, radius):
        target_rect = pygame.Rect(center, (0, 0)).inflate((radius * 2, radius * 2))
        shape_surf = pygame.Surface(target_rect.size, pygame.SRCALPHA)
        pygame.draw.circle(shape_surf, color, (radius, radius), radius)
        surface.blit(shape_surf, target_rect)

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually
    # It will create a window and render the environment
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Unset the dummy video driver to allow window creation
    os.environ.pop("SDL_VIDEODRIVER", None)
    pygame.display.init()
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Cell Membrane Defender")
    clock = pygame.time.Clock()
    
    terminated = False
    truncated = False
    total_reward = 0
    
    print("\n--- Manual Control ---")
    print(GameEnv.user_guide)
    print("R: Reset")
    print("----------------------\n")

    while not terminated and not truncated:
        movement = 0 # No-op
        space_held = 0
        shift_held = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
                print("--- Environment Reset ---")

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1

        action = [movement, space_held, shift_held]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(env.FPS)
        
        if terminated or truncated:
            print(f"Episode Finished. Final Score: {info['score']}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
            print(f"Terminated: {terminated}, Truncated: {truncated}")

    env.close()