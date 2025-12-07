import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move your blade. Press space to perform a quick slice."
    )

    game_description = (
        "Slice falling fruit with a virtual blade to achieve the highest score. "
        "Slice 20 fruits to win, but miss 5 and you lose!"
    )

    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.width = 640
        self.height = 400
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()
        
        # --- Visuals ---
        self.font_large = pygame.font.SysFont("Arial", 36, bold=True)
        self.font_small = pygame.font.SysFont("Arial", 24)
        
        self.COLOR_BG_1 = (15, 25, 40)
        self.COLOR_BG_2 = (30, 45, 65)
        self.COLOR_BLADE = (220, 255, 255)
        self.COLOR_TRAIL = (150, 220, 255)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_MISS = (255, 80, 80)
        
        self.FRUIT_COLORS = {
            "apple": (255, 60, 60),
            "orange": (255, 165, 0),
            "lemon": (255, 255, 100),
            "lime": (150, 255, 150),
        }
        self.fruit_types = list(self.FRUIT_COLORS.keys())

        # Initialize state variables
        self.blade_pos = None
        self.blade_trail = None
        self.slicing_timer = None
        self.fruits = None
        self.particles = None
        self.miss_indicators = None
        self.steps = None
        self.score = None
        self.missed_fruits = None
        self.sliced_fruits_stage = None
        self.sliced_fruits_total = None
        self.fruit_spawn_timer = None
        self.game_over = None
        
        # This is called here to set up the np_random object
        self.reset()
        
        # self.validate_implementation() # Temporarily disabled for initialization

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.blade_pos = pygame.Vector2(self.width // 2, self.height // 2)
        self.blade_trail = []
        self.slicing_timer = 0
        
        self.fruits = []
        self.particles = []
        self.miss_indicators = []
        
        self.steps = 0
        self.score = 0
        self.missed_fruits = 0
        self.sliced_fruits_stage = 0
        self.sliced_fruits_total = 0
        
        self.fruit_spawn_timer = 60 # Spawn first fruit after 2 seconds
        self.game_over = False
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        
        # --- Handle Input and Blade Movement ---
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        blade_prev_pos = pygame.Vector2(self.blade_pos)
        
        blade_speed = 15
        if movement == 1: self.blade_pos.y -= blade_speed
        elif movement == 2: self.blade_pos.y += blade_speed
        elif movement == 3: self.blade_pos.x -= blade_speed
        elif movement == 4: self.blade_pos.x += blade_speed
            
        self.blade_pos.x = np.clip(self.blade_pos.x, 0, self.width)
        self.blade_pos.y = np.clip(self.blade_pos.y, 0, self.height)
        
        self.blade_trail.append(pygame.Vector2(self.blade_pos))
        if len(self.blade_trail) > 10:
            self.blade_trail.pop(0)

        if space_held and self.slicing_timer <= 0:
            self.slicing_timer = 5 # Slice effect lasts for 5 frames
            # Sound: Sword_Slice.wav

        # --- Continuous Reward for Moving Towards Fruit ---
        if self.fruits:
            closest_fruit = min(self.fruits, key=lambda f: self.blade_pos.distance_to(f['pos']))
            dist_before = blade_prev_pos.distance_to(closest_fruit['pos'])
            dist_after = self.blade_pos.distance_to(closest_fruit['pos'])
            if dist_after < dist_before:
                reward += 0.01
            else:
                reward -= 0.001

        # --- Update Game State ---
        self._update_difficulty()
        self._update_spawner()
        
        newly_sliced_fruits, new_reward = self._update_fruits()
        reward += new_reward
        
        self._update_particles()
        self._update_miss_indicators()
        
        self.slicing_timer = max(0, self.slicing_timer - 1)

        # --- Check Termination Conditions ---
        terminated = False
        if self.sliced_fruits_stage >= 20: # Win
            reward += 10
            terminated = True
            self.game_over = True
        elif self.missed_fruits >= 5: # Loss
            reward -= 10
            terminated = True
            self.game_over = True
        elif self.steps >= 1800: # Timeout (60s @ 30fps)
            terminated = True
            self.game_over = True

        truncated = False # No truncation condition in this game
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _update_difficulty(self):
        # Increase speed every 20 fruits, spawn rate every 10
        self.fall_speed_multiplier = 1.0 + (self.sliced_fruits_total // 20) * 0.2
        self.spawn_rate = 60 - (self.sliced_fruits_total // 10) * 5
        self.spawn_rate = max(15, self.spawn_rate) # Cap at 4 spawns/sec

    def _update_spawner(self):
        self.fruit_spawn_timer -= 1
        if self.fruit_spawn_timer <= 0:
            self.fruit_spawn_timer = self.spawn_rate
            
            fruit_type = self.np_random.choice(self.fruit_types)
            radius = self.np_random.integers(20, 31)
            pos_x = self.np_random.uniform(radius, self.width - radius)
            
            self.fruits.append({
                'pos': pygame.Vector2(pos_x, -radius),
                'vel': pygame.Vector2(self.np_random.uniform(-1, 1), self.np_random.uniform(2.5, 4.0) * self.fall_speed_multiplier),
                'radius': radius,
                'color': self.FRUIT_COLORS[fruit_type],
                'wobble_angle': self.np_random.uniform(0, 2 * math.pi),
                'wobble_speed': self.np_random.uniform(0.05, 0.15)
            })

    def _update_fruits(self):
        reward = 0
        sliced_this_frame = []
        
        for fruit in self.fruits[:]:
            fruit['pos'] += fruit['vel']
            fruit['wobble_angle'] += fruit['wobble_speed']
            
            # Check for slice
            is_slicing = self.slicing_timer > 0
            if is_slicing and len(self.blade_trail) > 1:
                p1 = self.blade_trail[-2]
                p2 = self.blade_trail[-1]
                
                # Interpolate along the slice path for better collision detection
                for i in range(4):
                    lerp_pos = p1.lerp(p2, i / 3)
                    if lerp_pos.distance_to(fruit['pos']) < fruit['radius']:
                        sliced_this_frame.append(fruit)
                        self.fruits.remove(fruit)
                        reward += 1.0
                        self.score += 1
                        self.sliced_fruits_stage += 1
                        self.sliced_fruits_total += 1
                        self._create_slice_particles(fruit)
                        # Sound: Fruit_Squish.wav
                        break
            
            # Check for miss
            if fruit not in sliced_this_frame and fruit['pos'].y > self.height + fruit['radius']:
                self.fruits.remove(fruit)
                self.missed_fruits += 1
                reward -= 1.0
                self.miss_indicators.append({'pos': pygame.Vector2(fruit['pos'].x, self.height - 30), 'timer': 60})
                # Sound: Miss_Sound.wav
                
        return sliced_this_frame, reward

    def _create_slice_particles(self, fruit):
        # Create a splash of particles
        for _ in range(30):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(2, 7)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                'pos': pygame.Vector2(fruit['pos']),
                'vel': vel,
                'radius': self.np_random.uniform(2, 5),
                'color': fruit['color'],
                'lifetime': self.np_random.integers(20, 40)
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['vel'] *= 0.95 # Air resistance
            p['lifetime'] -= 1
            p['radius'] -= 0.1
            if p['lifetime'] <= 0 or p['radius'] <= 0:
                self.particles.remove(p)

    def _update_miss_indicators(self):
        for m in self.miss_indicators[:]:
            m['timer'] -= 1
            if m['timer'] <= 0:
                self.miss_indicators.remove(m)

    def _get_observation(self):
        # Draw background gradient
        for y in range(self.height):
            interp = y / self.height
            color = (
                self.COLOR_BG_1[0] * (1 - interp) + self.COLOR_BG_2[0] * interp,
                self.COLOR_BG_1[1] * (1 - interp) + self.COLOR_BG_2[1] * interp,
                self.COLOR_BG_1[2] * (1 - interp) + self.COLOR_BG_2[2] * interp,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.width, y))
        
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render fruits
        for fruit in self.fruits:
            pos = (int(fruit['pos'].x), int(fruit['pos'].y))
            radius = int(fruit['radius'])
            wobble_x = math.cos(fruit['wobble_angle']) * 2
            wobble_y = math.sin(fruit['wobble_angle'] * 2) * 2
            draw_pos = (pos[0] + int(wobble_x), pos[1] + int(wobble_y))
            pygame.gfxdraw.filled_circle(self.screen, draw_pos[0], draw_pos[1], radius, fruit['color'])
            pygame.gfxdraw.aacircle(self.screen, draw_pos[0], draw_pos[1], radius, fruit['color'])

        # Render particles
        for p in self.particles:
            pos = (int(p['pos'].x), int(p['pos'].y))
            radius = int(p['radius'])
            if radius > 0:
                alpha = int(255 * (p['lifetime'] / 40))
                color = (*p['color'], alpha)
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, color)

        # Render blade trail
        if len(self.blade_trail) > 1:
            points = [(int(p.x), int(p.y)) for p in self.blade_trail]
            for i in range(len(points) - 1):
                alpha = int(255 * (i / len(points)))
                width = int(8 * (i / len(points)))
                # Pygame's draw functions don't handle alpha well for width > 1, so we don't use it here.
                # A proper implementation would use a separate surface with SRCALPHA.
                color = self.COLOR_TRAIL 
                pygame.draw.line(self.screen, color, points[i], points[i+1], width)

        # Render blade cursor (glow effect)
        blade_pos_int = (int(self.blade_pos.x), int(self.blade_pos.y))
        for i in range(5, 0, -1):
            alpha = 100 - i * 20
            pygame.gfxdraw.filled_circle(self.screen, blade_pos_int[0], blade_pos_int[1], i * 2, (*self.COLOR_BLADE, alpha))
        pygame.gfxdraw.filled_circle(self.screen, blade_pos_int[0], blade_pos_int[1], 5, self.COLOR_BLADE)
        
        # Render miss indicators
        for m in self.miss_indicators:
            alpha = int(255 * (m['timer'] / 60))
            color = self.COLOR_MISS
            pos = (int(m['pos'].x), int(m['pos'].y))
            # Create a temporary surface for alpha blending
            s = pygame.Surface((30,30), pygame.SRCALPHA)
            pygame.draw.line(s, (*color, alpha), (0,0), (30,30), 5)
            pygame.draw.line(s, (*color, alpha), (30,0), (0,30), 5)
            self.screen.blit(s, (pos[0]-15, pos[1]-15))


    def _render_ui(self):
        # Score display
        score_text = self.font_large.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (15, 10))
        
        # Misses display
        miss_text = self.font_small.render(f"Misses: {self.missed_fruits} / 5", True, self.COLOR_TEXT)
        text_rect = miss_text.get_rect(topright=(self.width - 15, 15))
        self.screen.blit(miss_text, text_rect)
        
        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            if self.sliced_fruits_stage >= 20:
                msg = "STAGE CLEAR!"
            else:
                msg = "GAME OVER"
            
            end_text = self.font_large.render(msg, True, self.COLOR_TEXT)
            end_rect = end_text.get_rect(center=(self.width / 2, self.height / 2))
            self.screen.blit(end_text, end_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "missed_fruits": self.missed_fruits,
            "sliced_fruits": self.sliced_fruits_stage,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.height, self.width, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.height, self.width, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.height, self.width, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    # The environment itself runs headless. This is for human interaction.
    os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv(render_mode='rgb_array')
    env.validate_implementation()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.width, env.height))
    pygame.display.set_caption("Fruit Slicer")
    clock = pygame.time.Clock()
    
    terminated = False
    
    while not terminated:
        movement = 0 # No-op
        space_held = 0
        shift_held = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
        
        action = [movement, space_held, shift_held]
        
        obs, reward, term, trunc, info = env.step(action)
        terminated = term
        
        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Run at 30 FPS
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
            pygame.time.wait(3000) # Wait 3 seconds before closing

    env.close()