
# Generated: 2025-08-27T17:34:04.480311
# Source Brief: brief_01572.md
# Brief Index: 1572

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move the bucket. Catch the falling critters!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Catch falling critters in a bucket before they escape. The critters fall faster over time. Catch 25 to win, but let 5 escape and you lose."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.WIN_SCORE = 25
        self.MAX_ESCAPES = 5
        self.MAX_STEPS = 1800 # 60 seconds at 30fps

        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 72)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        # Colors
        self.COLOR_BG_TOP = (135, 206, 235)  # Sky Blue
        self.COLOR_BG_BOTTOM = (70, 130, 180) # Steel Blue
        self.COLOR_BUCKET = (139, 69, 19) # Saddle Brown
        self.COLOR_BUCKET_RIM = (160, 82, 45) # Sienna
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_TEXT_SHADOW = (0, 0, 0, 128)
        self.CRITTER_COLORS = [
            (255, 87, 51),   # Vermilion
            (255, 195, 0),   # Sunglow
            (51, 187, 255),  # Capri
            (102, 255, 102)  # Light Green
        ]
        self.HEART_COLOR = (255, 20, 20)

        # Game state attributes (initialized in reset)
        self.steps = None
        self.score = None
        self.escaped_critters = None
        self.game_over = None
        self.bucket_x = None
        self.critters = None
        self.particles = None
        self.critter_spawn_timer = None
        self.base_critter_speed = None
        self.last_bucket_dist = None
        self.np_random = None

        # Pre-render the background for efficiency
        self.background_surface = self._create_gradient_background()
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)
        
        self.steps = 0
        self.score = 0
        self.escaped_critters = 0
        self.game_over = False
        
        self.bucket_width = 80
        self.bucket_height = 40
        self.bucket_speed = 10
        self.bucket_x = self.WIDTH // 2 - self.bucket_width // 2
        
        self.critters = []
        self.particles = []
        self.critter_spawn_timer = self.np_random.integers(30, 60)
        self.base_critter_speed = 2.0
        self.last_bucket_dist = float('inf')
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Action Handling ---
        movement = action[0]  # 0-4: none/up/down/left/right
        
        prev_bucket_x = self.bucket_x
        if movement == 3:  # Left
            self.bucket_x -= self.bucket_speed
        elif movement == 4:  # Right
            self.bucket_x += self.bucket_speed
        
        # Clamp bucket position to screen bounds
        self.bucket_x = max(0, min(self.WIDTH - self.bucket_width, self.bucket_x))

        # --- Game Logic Update ---
        self._update_critters()
        self._update_particles()
        self._spawn_critters()

        # --- Collision and Event Detection ---
        reward = 0
        critters_caught_this_step = 0
        critters_escaped_this_step = 0

        bucket_rect = pygame.Rect(self.bucket_x, self.HEIGHT - self.bucket_height, self.bucket_width, self.bucket_height)
        
        for critter in self.critters[:]:
            critter_rect = pygame.Rect(critter['pos'][0] - critter['size']//2, critter['pos'][1] - critter['size']//2, critter['size'], critter['size'])
            
            # Catch condition
            if bucket_rect.colliderect(critter_rect):
                self.score += 1
                critters_caught_this_step += 1
                self._create_particles(critter['pos'], critter['color'])
                self.critters.remove(critter)
                # Sound effect: Catch success
                
                # Difficulty scaling
                if self.score > 0 and self.score % 5 == 0:
                    self.base_critter_speed += 0.2

            # Escape condition
            elif critter['pos'][1] > self.HEIGHT:
                self.escaped_critters += 1
                critters_escaped_this_step += 1
                self.critters.remove(critter)
                # Sound effect: Miss

        # --- Reward Calculation ---
        reward += critters_caught_this_step * 10
        reward -= critters_escaped_this_step * 10
        
        # Continuous reward for moving towards the nearest critter
        if self.critters:
            closest_critter = min(self.critters, key=lambda c: abs(c['pos'][0] - (self.bucket_x + self.bucket_width / 2)))
            current_dist = abs(closest_critter['pos'][0] - (self.bucket_x + self.bucket_width / 2))
            
            if self.last_bucket_dist is not None:
                if current_dist < self.last_bucket_dist:
                    reward += 0.1
                elif current_dist > self.last_bucket_dist:
                    reward -= 0.1
            self.last_bucket_dist = current_dist
        else:
            self.last_bucket_dist = float('inf')

        # --- Termination Check ---
        terminated = (self.score >= self.WIN_SCORE or 
                      self.escaped_critters >= self.MAX_ESCAPES or 
                      self.steps >= self.MAX_STEPS)
        
        if terminated and not self.game_over:
            self.game_over = True
            if self.score >= self.WIN_SCORE:
                reward += 100 # Win bonus
            elif self.escaped_critters >= self.MAX_ESCAPES:
                reward -= 100 # Loss penalty

        self.steps += 1
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _spawn_critters(self):
        self.critter_spawn_timer -= 1
        if self.critter_spawn_timer <= 0:
            critter_x = self.np_random.integers(20, self.WIDTH - 20)
            critter_size = self.np_random.integers(15, 25)
            critter_speed = self.base_critter_speed + self.np_random.uniform(-0.5, 0.5)
            critter_color = random.choice(self.CRITTER_COLORS)
            
            self.critters.append({
                'pos': [critter_x, -critter_size],
                'speed': max(1.0, critter_speed),
                'size': critter_size,
                'color': critter_color,
                'wobble_offset': self.np_random.uniform(0, 2 * math.pi),
                'wobble_speed': self.np_random.uniform(0.1, 0.3),
                'wobble_amount': self.np_random.uniform(5, 15)
            })
            
            spawn_delay = max(10, 60 - self.score * 1.5)
            self.critter_spawn_timer = self.np_random.integers(int(spawn_delay*0.8), int(spawn_delay*1.2))

    def _update_critters(self):
        for critter in self.critters:
            critter['pos'][1] += critter['speed']
            wobble = math.sin(self.steps * critter['wobble_speed'] + critter['wobble_offset']) * critter['wobble_amount']
            critter['pos'][0] += wobble * 0.1 # subtle horizontal movement

    def _update_particles(self):
        for p in self.particles[:]:
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
            else:
                p['pos'][0] += p['vel'][0]
                p['pos'][1] += p['vel'][1]
                p['vel'][1] += 0.1 # gravity
                p['radius'] -= 0.2

    def _create_particles(self, pos, color):
        # Sound effect: Poof
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'radius': self.np_random.uniform(3, 7),
                'color': color,
                'life': self.np_random.integers(15, 30)
            })

    def _get_observation(self):
        self.screen.blit(self.background_surface, (0, 0))
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "lives_left": self.MAX_ESCAPES - self.escaped_critters,
            "steps": self.steps,
        }

    def _create_gradient_background(self):
        bg = pygame.Surface((self.WIDTH, self.HEIGHT))
        for y in range(self.HEIGHT):
            color_ratio = y / self.HEIGHT
            r = self.COLOR_BG_TOP[0] * (1 - color_ratio) + self.COLOR_BG_BOTTOM[0] * color_ratio
            g = self.COLOR_BG_TOP[1] * (1 - color_ratio) + self.COLOR_BG_BOTTOM[1] * color_ratio
            b = self.COLOR_BG_TOP[2] * (1 - color_ratio) + self.COLOR_BG_BOTTOM[2] * color_ratio
            pygame.draw.line(bg, (int(r), int(g), int(b)), (0, y), (self.WIDTH, y))
        return bg

    def _render_game(self):
        # Draw critters
        for critter in self.critters:
            pos = (int(critter['pos'][0]), int(critter['pos'][1]))
            size = int(critter['size'])
            wobble = math.sin(self.steps * critter['wobble_speed'] + critter['wobble_offset']) * 2
            pygame.gfxdraw.filled_ellipse(self.screen, pos[0], pos[1], size // 2, int(size // 2 + wobble), critter['color'])
            pygame.gfxdraw.aaellipse(self.screen, pos[0], pos[1], size // 2, int(size // 2 + wobble), critter['color'])

        # Draw particles
        for p in self.particles:
            if p['radius'] > 0:
                pos = (int(p['pos'][0]), int(p['pos'][1]))
                radius = int(p['radius'])
                alpha = max(0, min(255, int(255 * (p['life'] / 20))))
                color = (*p['color'], alpha)
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, color)

        # Draw bucket
        bucket_rect = pygame.Rect(int(self.bucket_x), int(self.HEIGHT - self.bucket_height), int(self.bucket_width), int(self.bucket_height))
        pygame.draw.rect(self.screen, self.COLOR_BUCKET, bucket_rect, border_bottom_left_radius=5, border_bottom_right_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_BUCKET_RIM, (bucket_rect.x, bucket_rect.y, bucket_rect.width, 10))

    def _render_text(self, text, font, color, position, shadow=True):
        if shadow:
            text_surf_shadow = font.render(text, True, self.COLOR_TEXT_SHADOW)
            self.screen.blit(text_surf_shadow, (position[0] + 2, position[1] + 2))
        text_surf = font.render(text, True, color)
        self.screen.blit(text_surf, position)

    def _render_ui(self):
        # Score
        self._render_text(f"Score: {self.score}", self.font_medium, self.COLOR_TEXT, (10, 10))

        # Lives (Hearts)
        for i in range(self.MAX_ESCAPES - self.escaped_critters):
            heart_pos = (self.WIDTH - 30 - i * 35, 15)
            # Simple heart shape using polygons
            points = [
                (heart_pos[0], heart_pos[1] + 5),
                (heart_pos[0] - 12, heart_pos[1] - 5),
                (heart_pos[0] - 6, heart_pos[1] - 12),
                (heart_pos[0], heart_pos[1] - 5),
                (heart_pos[0] + 6, heart_pos[1] - 12),
                (heart_pos[0] + 12, heart_pos[1] - 5)
            ]
            pygame.draw.polygon(self.screen, self.HEART_COLOR, points)
            pygame.gfxdraw.aapolygon(self.screen, points, self.HEART_COLOR)

        # Game Over Message
        if self.game_over:
            s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            s.fill((0, 0, 0, 128))
            self.screen.blit(s, (0, 0))
            if self.score >= self.WIN_SCORE:
                msg = "You Win!"
            else:
                msg = "Game Over"
            
            self._render_text(msg, self.font_large, self.COLOR_TEXT, 
                              (self.WIDTH/2 - self.font_large.size(msg)[0]/2, self.HEIGHT/2 - 50))
            
            final_score_text = f"Final Score: {self.score}"
            self._render_text(final_score_text, self.font_medium, self.COLOR_TEXT,
                              (self.WIDTH/2 - self.font_medium.size(final_score_text)[0]/2, self.HEIGHT/2 + 20))

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    import os
    os.environ['SDL_VIDEODRIVER'] = 'x11' # Use 'x11' or 'dummy' for headless, 'windows' for windows
    
    env = GameEnv(render_mode="rgb_array")
    
    # To display the game, we need a different setup
    pygame.display.set_caption("Critter Catch")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    obs, info = env.reset()
    done = False
    
    # --- Manual Play Loop ---
    running = True
    while running:
        # Action defaults
        movement = 0 # no-op
        space_held = 0
        shift_held = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        if keys[pygame.K_r]: # Reset key
            obs, info = env.reset()
            done = False
            continue

        if not done:
            action = [movement, space_held, shift_held]
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        # --- Rendering ---
        # The observation is already a rendered frame
        # We just need to convert it back to a Pygame surface to display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(30) # Run at 30 FPS

    env.close()