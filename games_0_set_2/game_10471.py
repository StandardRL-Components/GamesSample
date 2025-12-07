import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:32:37.876626
# Source Brief: brief_00471.md
# Brief Index: 471
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the agent weaves colored threads to create
    barriers and portals, defending a central core against waves of nightmares.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = "Weave colored threads to create barriers and portals, defending a central core against waves of attacking nightmares."
    user_guide = "Controls: Use ↑↓←→ to weave barriers. Press space to activate portals of the selected color, and press shift to cycle through available thread colors."
    auto_advance = True

    # --- CONSTANTS ---
    WIDTH, HEIGHT = 640, 400
    CORE_POS = (WIDTH // 2, HEIGHT // 2)
    MAX_STEPS = 3000  # Approx 100 seconds at 30 FPS
    MAX_WAVES = 20

    # Colors
    COLOR_BG = (10, 5, 25)
    COLOR_CORE = (220, 220, 255)
    COLOR_NIGHTMARE = (200, 50, 200)
    COLOR_UI_TEXT = (220, 220, 220)
    THREAD_COLORS = {
        'red': (255, 50, 50),
        'blue': (50, 150, 255),
        'green': (50, 255, 100),
        'yellow': (255, 255, 50),
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gym Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 24)
        self.font_wave = pygame.font.Font(None, 36)

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.current_wave = 0
        self.core_health = 0
        self.nightmares = []
        self.barriers = []
        self.particles = []
        self.available_thread_types = []
        self.selected_thread_index = 0
        self.prev_space_held = False
        self.prev_shift_held = False

        self._generate_starfield()
        # self.reset() is called by the wrapper/runner, no need to call it here


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.current_wave = 0
        self.core_health = 100
        self.nightmares = []
        self.barriers = []
        self.particles = []
        self.available_thread_types = ['red', 'blue', 'green']
        self.selected_thread_index = 0
        self.prev_space_held = False
        self.prev_shift_held = False

        self._start_next_wave()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward_info = {"value": 0}

        self._handle_input(action, reward_info)
        self._update_nightmares(reward_info)
        self._update_barriers()
        self._update_particles()
        
        # Wave progression
        if not self.nightmares and not self.game_over:
            self.score += 5
            reward_info["value"] += 5
            self.current_wave += 1

            if self.current_wave >= self.MAX_WAVES:
                self.game_over = True
                self.score += 100
                reward_info["value"] += 100
            else:
                if self.current_wave == 4: # Unlock at start of wave 5
                    self.available_thread_types.append('yellow')
                    self.score += 2
                    reward_info["value"] += 2
                self._start_next_wave()

        # Termination conditions
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        if self.core_health <= 0 and not self.game_over:
            self.game_over = True
            terminated = True
            self.score -= 100
            reward_info["value"] -= 100
        
        truncated = False

        return (
            self._get_observation(),
            reward_info["value"],
            terminated,
            truncated,
            self._get_info()
        )

    # --- Game Logic ---

    def _start_next_wave(self):
        num_nightmares = 3 + self.current_wave
        base_health = 10 + self.current_wave * 2
        base_speed = 0.5 + self.current_wave * 0.05

        for _ in range(num_nightmares):
            side = self.np_random.integers(0, 4)
            if side == 0: # top
                pos = [self.np_random.uniform(0, self.WIDTH), -20]
            elif side == 1: # right
                pos = [self.WIDTH + 20, self.np_random.uniform(0, self.HEIGHT)]
            elif side == 2: # bottom
                pos = [self.np_random.uniform(0, self.WIDTH), self.HEIGHT + 20]
            else: # left
                pos = [-20, self.np_random.uniform(0, self.HEIGHT)]

            self.nightmares.append({
                'pos': np.array(pos, dtype=float),
                'health': base_health,
                'max_health': base_health,
                'speed': base_speed,
                'slow_timer': 0,
                'stun_timer': 0
            })

    def _handle_input(self, action, reward_info):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Weave barrier
        if movement != 0:
            direction_map = {1: 'up', 2: 'down', 3: 'left', 4: 'right'}
            direction = direction_map[movement]
            
            # Check if a barrier already exists in this direction
            if not any(b['dir'] == direction for b in self.barriers):
                color = self.available_thread_types[self.selected_thread_index]
                health = 100 if color != 'green' else 200 # Green barriers are stronger
                # Sound: WEAVE_THREAD
                self.barriers.append({'dir': direction, 'color': color, 'health': health, 'max_health': health, 'age': 0})
        
        # Activate portals (on press)
        if space_held and not self.prev_space_held:
            color_to_activate = self.available_thread_types[self.selected_thread_index]
            activated_any = False
            barriers_to_remove = []

            for barrier in self.barriers:
                if barrier['color'] == color_to_activate:
                    activated_any = True
                    barriers_to_remove.append(barrier)
                    self._activate_portal(barrier, reward_info)
            
            if activated_any:
                # Sound: PORTAL_ACTIVATE
                self.barriers = [b for b in self.barriers if b not in barriers_to_remove]

        # Cycle thread type (on press)
        if shift_held and not self.prev_shift_held:
            self.selected_thread_index = (self.selected_thread_index + 1) % len(self.available_thread_types)
            # Sound: UI_SWITCH

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

    def _activate_portal(self, barrier, reward_info):
        color = barrier['color']
        radius = 100
        
        if color == 'red':
            # Sound: EXPLOSION
            self._create_particles(self.CORE_POS, 50, self.THREAD_COLORS['red'], radius, 10)
            for n in self.nightmares:
                if np.linalg.norm(n['pos'] - self.CORE_POS) < radius:
                    n['health'] -= 50
        elif color == 'blue':
            # Sound: SLOW_FIELD
            self._create_particles(self.CORE_POS, 50, self.THREAD_COLORS['blue'], radius, 5)
            for n in self.nightmares:
                if np.linalg.norm(n['pos'] - self.CORE_POS) < radius:
                    n['slow_timer'] = 150 # 5 seconds at 30fps
        elif color == 'green':
            # Sound: HEAL_PULSE
            heal_amount = 20
            self.core_health = min(100, self.core_health + heal_amount)
            self._create_particles(self.CORE_POS, 30, self.THREAD_COLORS['green'], 50, 5)
        elif color == 'yellow':
            # Sound: STUN_WAVE
            self._create_particles(self.CORE_POS, 50, self.THREAD_COLORS['yellow'], radius, 8)
            for n in self.nightmares:
                if np.linalg.norm(n['pos'] - self.CORE_POS) < radius:
                    n['stun_timer'] = 90 # 3 seconds at 30fps

    def _update_nightmares(self, reward_info):
        alive_nightmares = []
        for n in self.nightmares:
            if n['stun_timer'] > 0:
                n['stun_timer'] -= 1
                alive_nightmares.append(n)
                continue

            direction = self.CORE_POS - n['pos']
            dist = np.linalg.norm(direction)
            
            if dist > 0:
                direction = direction / dist

            speed = n['speed']
            if n['slow_timer'] > 0:
                speed *= 0.3
                n['slow_timer'] -= 1
            
            n['pos'] += direction * speed

            # Collision with core
            if dist < 20:
                damage = 10
                self.core_health -= damage
                reward_info["value"] -= damage * 0.1 # -1 per hit
                self.score -= 1
                # Sound: CORE_DAMAGE
                self._create_particles(self.CORE_POS, 20, (255, 100, 100), 10, 5, is_core_hit=True)
                continue # Nightmare is destroyed

            # Collision with barriers
            collided = False
            for b in self.barriers:
                if self._check_nightmare_barrier_collision(n, b):
                    collided = True
                    b['health'] -= 1
                    if b['color'] == 'red':
                        n['health'] -= 1.5
                    elif b['color'] == 'blue':
                        n['slow_timer'] = max(n['slow_timer'], 10)
                    break
            
            if n['health'] > 0 and not collided:
                alive_nightmares.append(n)
            elif n['health'] <= 0:
                reward_info["value"] += 1
                self.score += 1
                # Sound: NIGHTMARE_DEATH
                self._create_particles(n['pos'], 30, self.COLOR_NIGHTMARE, 15, 3)

        self.nightmares = alive_nightmares

    def _check_nightmare_barrier_collision(self, nightmare, barrier):
        n_pos = nightmare['pos']
        n_radius = 8
        b_len = 150

        if barrier['dir'] == 'up':
            p1, p2 = (self.CORE_POS[0], self.CORE_POS[1]), (self.CORE_POS[0], self.CORE_POS[1] - b_len)
        elif barrier['dir'] == 'down':
            p1, p2 = (self.CORE_POS[0], self.CORE_POS[1]), (self.CORE_POS[0], self.CORE_POS[1] + b_len)
        elif barrier['dir'] == 'left':
            p1, p2 = (self.CORE_POS[0], self.CORE_POS[1]), (self.CORE_POS[0] - b_len, self.CORE_POS[1])
        else: # right
            p1, p2 = (self.CORE_POS[0], self.CORE_POS[1]), (self.CORE_POS[0] + b_len, self.CORE_POS[1])

        # Simple bounding box check
        if barrier['dir'] in ['up', 'down']:
            return abs(n_pos[0] - p1[0]) < n_radius and min(p1[1], p2[1]) < n_pos[1] < max(p1[1], p2[1])
        else: # left, right
            return abs(n_pos[1] - p1[1]) < n_radius and min(p1[0], p2[0]) < n_pos[0] < max(p1[0], p2[0])

    def _update_barriers(self):
        self.barriers = [b for b in self.barriers if b['health'] > 0]
        for b in self.barriers:
            b['age'] += 1

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['lifetime'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifetime'] -= 1

    # --- Rendering ---

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_starfield()
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._render_barriers()
        self._render_core()
        self._render_nightmares()
        self._render_particles()

    def _render_core(self):
        pulse = (math.sin(self.steps * 0.1) + 1) / 2
        
        # Outer glow
        glow_radius = int(20 + pulse * 5)
        glow_alpha = int(50 + pulse * 20)
        self._draw_glowing_circle(self.COLOR_CORE, self.CORE_POS, glow_radius, glow_alpha)

        # Inner core
        pygame.gfxdraw.filled_circle(self.screen, int(self.CORE_POS[0]), int(self.CORE_POS[1]), 15, self.COLOR_CORE)
        pygame.gfxdraw.aacircle(self.screen, int(self.CORE_POS[0]), int(self.CORE_POS[1]), 15, self.COLOR_CORE)

    def _render_barriers(self):
        b_len = 150
        for b in self.barriers:
            color = self.THREAD_COLORS[b['color']]
            health_ratio = b['health'] / b['max_health']
            
            if b['dir'] == 'up': p2 = (self.CORE_POS[0], self.CORE_POS[1] - b_len)
            elif b['dir'] == 'down': p2 = (self.CORE_POS[0], self.CORE_POS[1] + b_len)
            elif b['dir'] == 'left': p2 = (self.CORE_POS[0] - b_len, self.CORE_POS[1])
            else: p2 = (self.CORE_POS[0] + b_len, self.CORE_POS[1])

            # Draw glowing line
            width = int(2 + health_ratio * 4)
            self._draw_glowing_line(self.CORE_POS, p2, color, width)

            # Draw portal at the end
            pulse = (math.sin(self.steps * 0.2 + b['age'] * 0.1) + 1) / 2
            self._draw_glowing_circle(color, p2, int(8 + pulse * 4), int(100 + pulse * 50))
            pygame.gfxdraw.filled_circle(self.screen, int(p2[0]), int(p2[1]), 5, color)

    def _render_nightmares(self):
        for n in self.nightmares:
            pos = (int(n['pos'][0]), int(n['pos'][1]))
            color = list(self.COLOR_NIGHTMARE)
            if n['slow_timer'] > 0: color = self.THREAD_COLORS['blue']
            if n['stun_timer'] > 0: color = self.THREAD_COLORS['yellow']

            # Draw a simple triangle shape
            size = 10
            points = [
                (pos[0], pos[1] - size),
                (pos[0] - size / 1.5, pos[1] + size / 2),
                (pos[0] + size / 1.5, pos[1] + size / 2)
            ]
            pygame.gfxdraw.filled_trigon(self.screen, int(points[0][0]), int(points[0][1]), int(points[1][0]), int(points[1][1]), int(points[2][0]), int(points[2][1]), color)
            pygame.gfxdraw.aatrigon(self.screen, int(points[0][0]), int(points[0][1]), int(points[1][0]), int(points[1][1]), int(points[2][0]), int(points[2][1]), color)

    def _render_particles(self):
        for p in self.particles:
            alpha = p['lifetime'] / p['max_lifetime']
            color = (*p['color'], int(alpha * 255))
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], p['size'], color)

    def _render_ui(self):
        # Wave number
        wave_text = self.font_wave.render(f"Wave {self.current_wave + 1}/{self.MAX_WAVES}", True, self.COLOR_UI_TEXT)
        self.screen.blit(wave_text, (10, 10))

        # Core Health
        health_text = self.font_ui.render(f"Core Integrity: {int(self.core_health)}%", True, self.COLOR_UI_TEXT)
        self.screen.blit(health_text, (self.WIDTH - health_text.get_width() - 10, 10))
        
        # Score
        score_text = self.font_ui.render(f"Score: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 35))

        # Thread selector
        num_types = len(self.available_thread_types)
        box_size = 30
        padding = 10
        total_width = num_types * box_size + (num_types - 1) * padding
        start_x = self.CORE_POS[0] - total_width / 2

        for i, color_name in enumerate(self.available_thread_types):
            x = start_x + i * (box_size + padding)
            y = self.HEIGHT - 45
            rect = pygame.Rect(x, y, box_size, box_size)
            
            color = self.THREAD_COLORS[color_name]
            pygame.draw.rect(self.screen, color, rect, border_radius=5)

            if i == self.selected_thread_index:
                pygame.draw.rect(self.screen, (255, 255, 255), rect, 3, border_radius=5)

    def _render_starfield(self):
        for x, y, b in self.starfield:
            self.screen.set_at((x, y), (b, b, b))

    # --- Helper & Utility Methods ---

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.current_wave + 1,
            "core_health": self.core_health,
            "nightmares_remaining": len(self.nightmares)
        }

    def _generate_starfield(self):
        self.starfield = []
        # Check if np_random exists, otherwise create a temporary one
        if not hasattr(self, 'np_random'):
            self.np_random, _ = gym.utils.seeding.np_random()
            
        for _ in range(200):
            x = self.np_random.integers(0, self.WIDTH)
            y = self.np_random.integers(0, self.HEIGHT)
            brightness = self.np_random.integers(20, 80)
            self.starfield.append((x, y, brightness))

    def _create_particles(self, pos, count, color, max_radius, max_speed, is_core_hit=False):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, max_speed)
            radius = self.np_random.uniform(0, max_radius)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            
            start_pos = pos
            if not is_core_hit:
                 start_pos = pos + np.array([math.cos(angle) * radius, math.sin(angle) * radius])

            self.particles.append({
                'pos': start_pos.copy(),
                'vel': vel,
                'lifetime': self.np_random.integers(15, 30),
                'max_lifetime': 30,
                'color': color,
                'size': self.np_random.integers(1, 4)
            })

    def _draw_glowing_circle(self, color, center, radius, alpha):
        surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(surf, (*color, alpha), (radius, radius), radius)
        self.screen.blit(surf, (center[0] - radius, center[1] - radius), special_flags=pygame.BLEND_RGBA_ADD)

    def _draw_glowing_line(self, p1, p2, color, width):
        for i in range(3):
            alpha = 150 / (i + 1)
            current_width = width + i * 4
            line_color = (*color, alpha)
            
            # Create a surface for the line to handle alpha correctly
            line_rect = pygame.Rect(min(p1[0], p2[0]) - current_width, min(p1[1], p2[1]) - current_width,
                                    abs(p1[0] - p2[0]) + 2 * current_width, abs(p1[1] - p2[1]) + 2 * current_width)
            line_surf = pygame.Surface(line_rect.size, pygame.SRCALPHA)
            
            # Adjust points to be relative to the new surface
            rel_p1 = (p1[0] - line_rect.left, p1[1] - line_rect.top)
            rel_p2 = (p2[0] - line_rect.left, p2[1] - line_rect.top)
            
            pygame.draw.line(line_surf, line_color, rel_p1, rel_p2, int(current_width))
            self.screen.blit(line_surf, line_rect.topleft, special_flags=pygame.BLEND_RGBA_ADD)

        pygame.draw.line(self.screen, color, p1, p2, int(width / 2))

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
        print("Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc is False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # Make sure to remove the dummy video driver for manual play
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup a window to display the game
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Dream Weaver")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        # --- Human Input ---
        keys = pygame.key.get_pressed()
        
        movement = 0 # none
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Rendering ---
        # The observation is already a rendered frame, so we just display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("--- RESETTING ---")
                obs, info = env.reset()
                total_reward = 0

        if terminated or truncated:
            print(f"Episode finished. Final Score: {info['score']}, Total Reward: {total_reward}")
            print("Press 'R' to restart.")

        clock.tick(30) # Cap FPS
        
    env.close()