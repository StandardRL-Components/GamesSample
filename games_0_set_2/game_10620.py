import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:40:38.213425
# Source Brief: brief_00620.md
# Brief Index: 620
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Grow a vine towards the sunlight, absorbing nutrients to change your color and avoiding dangerous predators. "
        "Unlock the ability to flip gravity to navigate the treacherous jungle."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to direct growth. Hold space to grow rapidly along the current gravity axis. "
        "Press shift to flip gravity (once unlocked)."
    )
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 2000
        self.SUNLIGHT_ZONE_HEIGHT = 20
        self.VINE_GROWTH_SPEED = 8
        self.VINE_SEGMENT_RADIUS = 5
        self.PREDATOR_BASE_SPEED = 1.0
        self.PREDATOR_MAX_SPEED = 3.0
        self.PREDATOR_SPEED_INCREASE = 0.2

        # Colors
        self.COLOR_BG = (26, 42, 26) # Dark jungle green
        self.COLOR_SUNLIGHT = (255, 249, 148)
        self.COLOR_VINE_BASE = (134, 194, 50)
        self.COLOR_VINE_HEAD = (210, 255, 128)
        self.COLOR_PREDATOR = (230, 57, 70)
        self.NUTRIENT_COLORS = [
            (69, 123, 157),  # Blue
            (252, 163, 17),   # Orange
            (168, 218, 220), # Cyan
        ]
        self.COLOR_UI_TEXT = (241, 250, 238)

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.large_font = pygame.font.Font(None, 36)
        
        # Initialize state variables
        self._initialize_state()
        
    
    def _initialize_state(self):
        """Initializes or resets all game state variables."""
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        
        # Player/Vine state
        start_pos = (self.WIDTH // 2, self.HEIGHT - self.VINE_SEGMENT_RADIUS)
        self.vine_segments = [{'pos': start_pos, 'color': self.COLOR_VINE_BASE, 'radius': self.VINE_SEGMENT_RADIUS}]
        self.vine_head_color = self.COLOR_VINE_BASE
        self.gravity_up = True
        self.gravity_flip_unlocked = False
        self.last_shift_press = False

        # Game entities
        self.predators = []
        self.nutrients = []
        self.particles = []
        
        # Background elements
        self.bg_elements = self._generate_background()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._initialize_state()
        
        # Initial game setup
        self._spawn_predator()
        for _ in range(5):
            self._spawn_nutrient()
        
        obs = self._get_observation()
        info = self._get_info()
        return obs, info
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement_action = action[0]
        space_held = action[1] == 1
        shift_held = action[2] == 1
        
        reward = -0.01  # Small penalty for existing
        self.steps += 1
        
        # --- Handle Actions ---
        self._handle_gravity_flip(shift_held)
        self._handle_growth(movement_action, space_held)
        
        # --- Update Game World ---
        self._update_predators()
        self._update_particles()
        
        # --- Check for Events & Collisions ---
        vine_head_pos = self.vine_segments[-1]['pos']
        
        # Nutrient collision
        nutrient_reward, absorbed = self._check_nutrient_collision(vine_head_pos)
        if absorbed:
            reward += nutrient_reward
            self.score += 1

        # Predator collision
        predator_reward, collided = self._check_predator_collision(vine_head_pos)
        if collided:
            reward += predator_reward
            self.game_over = True
            
        # Boundary collision
        if not (0 <= vine_head_pos[0] < self.WIDTH and 0 <= vine_head_pos[1] < self.HEIGHT):
            reward -= 10
            self.game_over = True

        # --- Check for Win Condition ---
        if vine_head_pos[1] < self.SUNLIGHT_ZONE_HEIGHT:
            reward += 100
            self.score += 100
            self.game_over = True

        # --- Progression ---
        self._update_progression()

        # --- Termination ---
        terminated = self.game_over
        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            self.game_over = True
        
        obs = self._get_observation()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    # --- Action Handling ---

    def _handle_gravity_flip(self, shift_held):
        if self.gravity_flip_unlocked and shift_held and not self.last_shift_press:
            self.gravity_up = not self.gravity_up
            self._create_particles(self.vine_segments[-1]['pos'], (200, 200, 255), 20, 2.0)
        self.last_shift_press = shift_held

    def _handle_growth(self, movement_action, space_held):
        direction = None
        if space_held:
            direction = (0, -1) if self.gravity_up else (0, 1)
        elif movement_action != 0:
            dirs = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
            direction = dirs[movement_action]
        
        if direction:
            old_head = self.vine_segments[-1]
            new_pos = (
                old_head['pos'][0] + direction[0] * self.VINE_GROWTH_SPEED,
                old_head['pos'][1] + direction[1] * self.VINE_GROWTH_SPEED
            )
            self.vine_segments.append({'pos': new_pos, 'color': self.vine_head_color, 'radius': self.VINE_SEGMENT_RADIUS})
            return True
        return False

    # --- Collision and Event Checks ---

    def _check_nutrient_collision(self, head_pos):
        for nutrient in self.nutrients[:]:
            dist = math.hypot(head_pos[0] - nutrient['pos'][0], head_pos[1] - nutrient['pos'][1])
            if dist < self.VINE_SEGMENT_RADIUS + nutrient['radius']:
                if self.vine_head_color == self.COLOR_VINE_BASE or self.vine_head_color == nutrient['color']:
                    self.vine_head_color = nutrient['color']
                    # Recolor last few segments
                    for seg in self.vine_segments[-min(5, len(self.vine_segments)):]:
                        seg['color'] = nutrient['color']
                    
                    self._create_particles(nutrient['pos'], nutrient['color'], 30, 1.5)
                    self.nutrients.remove(nutrient)
                    self._spawn_nutrient()
                    return 1.0, True
        return 0.0, False

    def _check_predator_collision(self, head_pos):
        for predator in self.predators:
            dist = math.hypot(head_pos[0] - predator['pos'][0], head_pos[1] - predator['pos'][1])
            if dist < self.VINE_SEGMENT_RADIUS + predator['radius']:
                self._create_particles(head_pos, self.COLOR_PREDATOR, 50, 3.0)
                return -10.0, True
        return 0.0, False

    # --- Game State Updates ---

    def _update_predators(self):
        for p in self.predators:
            p['pos'] = (p['pos'][0] + p['vel'][0], p['pos'][1] + p['vel'][1])
            if p['pos'][0] < p['radius'] or p['pos'][0] > self.WIDTH - p['radius']:
                p['vel'] = (-p['vel'][0], p['vel'][1])
            if p['pos'][1] < p['radius'] or p['pos'][1] > self.HEIGHT - p['radius']:
                p['vel'] = (p['vel'][0], -p['vel'][1])

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] = (p['pos'][0] + p['vel'][0], p['pos'][1] + p['vel'][1])
            p['life'] -= 1

    def _update_progression(self):
        if not self.gravity_flip_unlocked and self.steps >= 100:
            self.gravity_flip_unlocked = True

        if self.steps > 0 and self.steps % 200 == 0:
            self._spawn_predator()
            speed_mult = 1.0 + (self.steps // 200) * self.PREDATOR_SPEED_INCREASE
            new_speed = min(self.PREDATOR_BASE_SPEED * speed_mult, self.PREDATOR_MAX_SPEED)
            for p in self.predators:
                current_speed = math.hypot(p['vel'][0], p['vel'][1])
                if current_speed > 0:
                    scale = new_speed / current_speed
                    p['vel'] = (p['vel'][0] * scale, p['vel'][1] * scale)
    
    # --- Spawning Logic ---
    
    def _spawn_predator(self):
        radius = random.randint(10, 15)
        side = random.choice(['top', 'bottom', 'left', 'right'])
        if side == 'left':
            pos = (radius, random.randint(radius, self.HEIGHT - radius))
            vel = (self.PREDATOR_BASE_SPEED, 0)
        elif side == 'right':
            pos = (self.WIDTH - radius, random.randint(radius, self.HEIGHT - radius))
            vel = (-self.PREDATOR_BASE_SPEED, 0)
        elif side == 'top':
            pos = (random.randint(radius, self.WIDTH - radius), radius)
            vel = (0, self.PREDATOR_BASE_SPEED)
        else: # bottom
            pos = (random.randint(radius, self.WIDTH - radius), self.HEIGHT - radius)
            vel = (0, -self.PREDATOR_BASE_SPEED)

        pattern = random.choice(['linear', 'bounce'])
        self.predators.append({'pos': pos, 'vel': vel, 'radius': radius, 'pattern': pattern})

    def _spawn_nutrient(self):
        pos = (
            random.randint(20, self.WIDTH - 20),
            random.randint(self.SUNLIGHT_ZONE_HEIGHT + 20, self.HEIGHT - 20)
        )
        color = random.choice(self.NUTRIENT_COLORS)
        radius = random.randint(5, 8)
        self.nutrients.append({'pos': pos, 'color': color, 'radius': radius})

    def _create_particles(self, pos, color, count, speed_scale):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(0.5, 1.5) * speed_scale
            vel = (math.cos(angle) * speed, math.sin(angle) * speed)
            self.particles.append({'pos': pos, 'vel': vel, 'life': random.randint(10, 25), 'color': color})

    def _generate_background(self):
        elements = []
        for _ in range(30):
            elements.append({
                'pos': (random.randint(0, self.WIDTH), random.randint(0, self.HEIGHT)),
                'radius': random.randint(50, 150),
                'color': (
                    self.COLOR_BG[0] + random.randint(5, 15),
                    self.COLOR_BG[1] + random.randint(5, 15),
                    self.COLOR_BG[2] + random.randint(5, 15)
                )
            })
        return elements

    # --- Rendering ---

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._render_background()
        self._render_sunlight_zone()
        self._render_nutrients()
        self._render_vine()
        self._render_predators()
        self._render_particles()

    def _render_background(self):
        for el in self.bg_elements:
            self._draw_glow_circle(el['pos'], el['radius'], el['color'], 0.5)

    def _render_sunlight_zone(self):
        surf = pygame.Surface((self.WIDTH, self.SUNLIGHT_ZONE_HEIGHT), pygame.SRCALPHA)
        for i in range(self.SUNLIGHT_ZONE_HEIGHT):
            alpha = 150 - int((i / self.SUNLIGHT_ZONE_HEIGHT) * 120)
            pygame.draw.line(surf, self.COLOR_SUNLIGHT + (alpha,), (0, i), (self.WIDTH, i))
        self.screen.blit(surf, (0, 0))

    def _render_vine(self):
        if not self.vine_segments: return
        
        # Draw body
        for i in range(len(self.vine_segments) - 1):
            p1 = self.vine_segments[i]
            p2 = self.vine_segments[i+1]
            pygame.draw.line(self.screen, p1['color'], (int(p1['pos'][0]), int(p1['pos'][1])), (int(p2['pos'][0]), int(p2['pos'][1])), int(p1['radius'] * 1.5))
        
        # Draw segments as circles on top
        for seg in self.vine_segments:
            pygame.gfxdraw.aacircle(self.screen, int(seg['pos'][0]), int(seg['pos'][1]), seg['radius'], seg['color'])
            pygame.gfxdraw.filled_circle(self.screen, int(seg['pos'][0]), int(seg['pos'][1]), seg['radius'], seg['color'])

        # Draw head glow
        head = self.vine_segments[-1]
        self._draw_glow_circle(head['pos'], head['radius'] + 2, self.COLOR_VINE_HEAD, 0.5)

    def _render_predators(self):
        for p in self.predators:
            self._draw_glow_circle(p['pos'], p['radius'], self.COLOR_PREDATOR, 0.8)
            pygame.gfxdraw.aacircle(self.screen, int(p['pos'][0]), int(p['pos'][1]), p['radius'], self.COLOR_PREDATOR)
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), p['radius'], self.COLOR_PREDATOR)
            
    def _render_nutrients(self):
        for n in self.nutrients:
            pulse = (math.sin(self.steps * 0.1 + n['pos'][0]) + 1) / 2
            radius = n['radius'] + int(pulse * 2)
            self._draw_glow_circle(n['pos'], radius, n['color'], 0.7)
            pygame.gfxdraw.aacircle(self.screen, int(n['pos'][0]), int(n['pos'][1]), radius, n['color'])
            pygame.gfxdraw.filled_circle(self.screen, int(n['pos'][0]), int(n['pos'][1]), radius, n['color'])

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['life'] / 25.0))
            if alpha > 0:
                color = p['color'] + (alpha,)
                surf = pygame.Surface((4, 4), pygame.SRCALPHA)
                pygame.draw.circle(surf, color, (2, 2), 2)
                self.screen.blit(surf, (int(p['pos'][0])-2, int(p['pos'][1])-2), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_ui(self):
        # Score
        score_text = self.font.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Vine Length
        length_text = self.font.render(f"LENGTH: {len(self.vine_segments)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(length_text, (self.WIDTH - length_text.get_width() - 10, self.HEIGHT - length_text.get_height() - 10))

        # Gravity Indicator
        arrow_char = "↑" if self.gravity_up else "↓"
        color = (100, 200, 255) if self.gravity_flip_unlocked else (100, 100, 100)
        grav_text = self.large_font.render(f"GRAVITY: {arrow_char}", True, color)
        self.screen.blit(grav_text, (self.WIDTH // 2 - grav_text.get_width() // 2, 10))

        if not self.gravity_flip_unlocked and self.steps > 50 and self.steps < 100:
             unlock_prog = (self.steps - 50) / 50.0
             alpha = int(unlock_prog * 255)
             unlock_text = self.font.render("Gravity Flip charging...", True, self.COLOR_UI_TEXT)
             unlock_text.set_alpha(alpha)
             self.screen.blit(unlock_text, (self.WIDTH // 2 - unlock_text.get_width() // 2, 40))
        elif self.gravity_flip_unlocked and self.steps < 200:
             alpha = max(0, 255 - (self.steps - 100) * 2)
             unlock_text = self.large_font.render("Gravity Flip Unlocked! (SHIFT)", True, (200,255,200))
             unlock_text.set_alpha(alpha)
             self.screen.blit(unlock_text, (self.WIDTH // 2 - unlock_text.get_width() // 2, 40))


    def _draw_glow_circle(self, pos, radius, color, intensity=0.5):
        glow_surf = pygame.Surface((radius * 4, radius * 4), pygame.SRCALPHA)
        for i in range(4):
            alpha = int(80 * intensity * ((4 - i) / 4.0))
            pygame.gfxdraw.filled_circle(
                glow_surf,
                int(radius * 2), int(radius * 2),
                int(radius + i * 2),
                color + (alpha,)
            )
        self.screen.blit(glow_surf, (int(pos[0] - radius*2), int(pos[1] - radius*2)), special_flags=pygame.BLEND_RGBA_ADD)

    # --- Gymnasium and Info ---

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "vine_length": len(self.vine_segments),
            "gravity_up": self.gravity_up,
            "gravity_flip_unlocked": self.gravity_flip_unlocked,
        }
    
    def close(self):
        pygame.quit()
        

if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Switch to a visible driver for manual play
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    pygame.quit()
    pygame.init()

    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Vine Climber")
    clock = pygame.time.Clock()
    
    terminated = False
    truncated = False
    total_reward = 0
    
    while not terminated and not truncated:
        # --- Action mapping for human play ---
        movement = 0 # 0=none
        space = 0 # 0=released
        shift = 0 # 0=released
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
        
        action = [movement, space, shift]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
                terminated = False
                truncated = False

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Rendering ---
        # The observation is already a rendered frame, so we just blit it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.FPS)

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
            
    env.close()