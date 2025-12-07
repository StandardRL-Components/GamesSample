import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A Gymnasium environment for a fast-paced arcade game, "Star Catcher".
    The player controls a basket at the bottom of the screen and must catch
    falling stars. Catching stars near the edge of the basket provides a score
    bonus. The game ends by catching enough stars (win) or missing too many (loss).
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use ← and → arrow keys to move the basket."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Catch falling stars in a moving basket. Risky catches near the edge of the basket yield bonus points."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        """Initializes the Star Catcher environment."""
        super().__init__()

        self.screen_width = 640
        self.screen_height = 400

        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.screen_height, self.screen_width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 36)
        self.font_msg = pygame.font.Font(None, 72)

        # Colors
        self.COLOR_BG = (15, 20, 40)
        self.COLOR_BASKET = (230, 60, 60)
        self.COLOR_STAR = (255, 255, 240)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_CATCH_GOOD = (80, 255, 80)
        self.COLOR_CATCH_BONUS = (255, 255, 80)
        self.COLOR_MISS = (255, 0, 0)

        # Game constants
        self.WIN_SCORE = 25
        self.MAX_LIVES = 5
        self.MAX_STEPS = 1000
        self.INITIAL_STAR_SPEED = 2.0
        self.BASKET_SPEED = 10.0

        # Initialize state variables
        self.basket_x = 0
        self.basket_width = 0
        self.basket_height = 0
        self.stars = []
        self.particles = []
        self.score = 0
        self.lives = 0
        self.steps = 0
        self.stars_caught = 0
        self.stars_caught_since_speed_increase = 0
        self.star_fall_speed = 0
        self.game_over = False
        self.win_message = ""
        self.screen_flash_timer = 0
        
        # Initial reset to populate state
        self.reset()
        
        # Validate implementation after initialization
        # self.validate_implementation() # Commented out for submission

    def reset(self, seed=None, options=None):
        """Resets the game to its initial state."""
        super().reset(seed=seed)
        # Note: The 'seed' argument is not used for 'random' module.
        # For reproducibility, one would need to call random.seed(seed)
        # but we keep it simple as per the original code.

        self.basket_width = 80
        self.basket_height = 20
        self.basket_x = self.screen_width / 2 - self.basket_width / 2

        self.stars = []
        self.particles = []
        self.score = 0
        self.lives = self.MAX_LIVES
        self.steps = 0
        self.stars_caught = 0
        self.stars_caught_since_speed_increase = 0
        self.star_fall_speed = self.INITIAL_STAR_SPEED
        self.game_over = False
        self.win_message = ""
        self.screen_flash_timer = 0

        self._spawn_star()

        return self._get_observation(), self._get_info()

    def step(self, action):
        """
        Advances the game by one timestep.
        Handles player input, updates game objects, and calculates rewards.
        """
        reward = 0.0
        
        if self.game_over:
            # If the game is over, no actions should have an effect.
            # We just return the final state.
            terminated = True
            return (
                self._get_observation(),
                0.0,
                terminated,
                False,
                self._get_info()
            )

        self.steps += 1
        
        # --- Action Handling ---
        movement = action[0]  # 0-4: none/up/down/left/right
        
        prev_basket_x = self.basket_x
        if movement == 3:  # Left
            self.basket_x -= self.BASKET_SPEED
        elif movement == 4:  # Right
            self.basket_x += self.BASKET_SPEED
        
        # Clamp basket to screen bounds
        self.basket_x = max(0, min(self.screen_width - self.basket_width, self.basket_x))

        # --- Continuous Reward ---
        if self.stars:
            # Reward for moving towards the lowest star
            lowest_star = min(self.stars, key=lambda s: -s['y'])
            basket_center = self.basket_x + self.basket_width / 2
            
            dist_before = abs(lowest_star['x'] - prev_basket_x - self.basket_width / 2)
            dist_after = abs(lowest_star['x'] - self.basket_x - self.basket_width / 2)
            
            if dist_after < dist_before:
                reward += 0.01 # Small reward for correct movement
            elif dist_after > dist_before:
                reward -= 0.01 # Small penalty for incorrect movement

        # --- Game Logic Update ---
        self._update_stars()
        self._update_particles()
        if self.screen_flash_timer > 0:
            self.screen_flash_timer -= 1

        # --- Star Catch/Miss Logic ---
        stars_to_remove = []
        for star in self.stars:
            basket_rect = pygame.Rect(self.basket_x, self.screen_height - self.basket_height, self.basket_width, self.basket_height)
            star_rect = pygame.Rect(star['x'] - 2, star['y'] - 2, 4, 4)

            if basket_rect.colliderect(star_rect):
                # Star caught
                catch_pos_ratio = abs((star['x'] - self.basket_x) / self.basket_width - 0.5) * 2
                
                if catch_pos_ratio > 0.7: # Risky catch (outer 15% on each side)
                    self.score += 2
                    reward += 2.0
                    self._create_particles(star['x'], star['y'], self.COLOR_CATCH_BONUS, 30)
                    # Sound: sfx_catch_bonus.wav
                else:
                    self.score += 1
                    reward += 1.0
                    self._create_particles(star['x'], star['y'], self.COLOR_CATCH_GOOD, 20)
                    # Sound: sfx_catch_normal.wav
                
                self.stars_caught += 1
                self.stars_caught_since_speed_increase += 1
                stars_to_remove.append(star)
                
                if self.stars_caught_since_speed_increase >= 5:
                    self.star_fall_speed += 0.2
                    self.stars_caught_since_speed_increase = 0

            elif star['y'] >= self.screen_height:
                # Star missed
                self.lives -= 1
                reward -= 1.0
                stars_to_remove.append(star)
                self.screen_flash_timer = 5 # Flash screen for 5 frames
                # Sound: sfx_miss.wav

        # Remove caught/missed stars
        if stars_to_remove:
            self.stars = [s for s in self.stars if s not in stars_to_remove]
            self._spawn_star()
        
        # --- Termination Check ---
        terminated = False
        truncated = False
        if self.stars_caught >= self.WIN_SCORE:
            terminated = True
            reward += 100.0  # Large win reward
            self.win_message = "YOU WIN!"
            # Sound: sfx_win.wav
        elif self.lives <= 0:
            terminated = True
            reward -= 100.0  # Large loss penalty
            self.win_message = "GAME OVER"
            # Sound: sfx_lose.wav
        elif self.steps >= self.MAX_STEPS:
            truncated = True # Episode ends due to time limit
            
        if terminated or truncated:
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _spawn_star(self):
        """Creates a new star at a random horizontal position at the top of the screen."""
        star = {
            'x': random.uniform(20, self.screen_width - 20),
            'y': 0,
            'trail': []
        }
        self.stars.append(star)

    def _update_stars(self):
        """Updates the position and trail of each star."""
        for star in self.stars:
            # Update trail
            star['trail'].append((star['x'], star['y']))
            if len(star['trail']) > 10:
                star['trail'].pop(0)
            
            # Move star
            star['y'] += self.star_fall_speed
            
    def _create_particles(self, x, y, color, count):
        """Generates a burst of particles for visual feedback."""
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            particle = {
                'x': x,
                'y': y,
                'vx': math.cos(angle) * speed,
                'vy': math.sin(angle) * speed,
                'life': random.randint(20, 40),
                'color': color
            }
            self.particles.append(particle)
            
    def _update_particles(self):
        """Updates the position and lifetime of each particle."""
        particles_to_keep = []
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['life'] -= 1
            if p['life'] > 0:
                particles_to_keep.append(p)
        self.particles = particles_to_keep
        
    def _get_observation(self):
        """Renders the current game state to a numpy array."""
        # --- Background ---
        self.screen.fill(self.COLOR_BG)

        # --- Render Particles ---
        for p in self.particles:
            alpha = int(255 * (p['life'] / 40))
            color = (*p['color'], alpha)
            temp_surf = pygame.Surface((3, 3), pygame.SRCALPHA)
            pygame.draw.rect(temp_surf, color, (0, 0, 3, 3))
            self.screen.blit(temp_surf, (int(p['x']), int(p['y'])))

        # --- Render Stars and Trails ---
        for star in self.stars:
            # Trail
            if star['trail']:
                for i, pos in enumerate(star['trail']):
                    alpha = int(150 * (i / len(star['trail'])))
                    pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), 2, (*self.COLOR_STAR, alpha))
            # Star
            pygame.gfxdraw.filled_circle(self.screen, int(star['x']), int(star['y']), 4, self.COLOR_STAR)
            pygame.gfxdraw.aacircle(self.screen, int(star['x']), int(star['y']), 4, self.COLOR_STAR)

        # --- Render Basket ---
        basket_rect = pygame.Rect(
            int(self.basket_x), 
            int(self.screen_height - self.basket_height), 
            int(self.basket_width), 
            int(self.basket_height)
        )
        pygame.draw.rect(self.screen, self.COLOR_BASKET, basket_rect, border_radius=3)
        
        # --- Screen Flash on Miss ---
        if self.screen_flash_timer > 0:
            flash_alpha = int(100 * (self.screen_flash_timer / 5))
            flash_surface = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
            flash_surface.fill((*self.COLOR_MISS, flash_alpha))
            self.screen.blit(flash_surface, (0,0))
            
        # --- Render UI ---
        score_text = self.font_ui.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        lives_text = self.font_ui.render(f"Lives: {self.lives}", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (self.screen_width - lives_text.get_width() - 10, 10))

        # --- Render Game Over Message ---
        if self.game_over:
            msg_surf = self.font_msg.render(self.win_message, True, self.COLOR_TEXT)
            msg_rect = msg_surf.get_rect(center=(self.screen_width / 2, self.screen_height / 2))
            self.screen.blit(msg_surf, msg_rect)

        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        """Returns a dictionary with auxiliary diagnostic information."""
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "stars_caught": self.stars_caught,
            "star_speed": self.star_fall_speed,
        }
        
    def close(self):
        """Cleans up Pygame resources."""
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
        assert test_obs.shape == (self.screen_height, self.screen_width, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.screen_height, self.screen_width, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.screen_height, self.screen_width, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# This block allows the game to be run and played directly for testing
if __name__ == '__main__':
    # This part of the code will not be run in the evaluation environment,
    # so we can use a regular pygame display here.
    os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv(render_mode="rgb_array")
    
    # Use a different screen for human rendering
    human_screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    pygame.display.set_caption("Star Catcher")
    
    obs, info = env.reset()
    done = False
    
    # Game loop
    running = True
    while running:
        action = np.array([0, 0, 0]) # Default no-op action
        
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Keyboard input
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        if keys[pygame.K_r]: # Press 'R' to reset
            obs, info = env.reset()
            done = False
            
        if keys[pygame.K_q]: # Press 'Q' to quit
            running = False

        # Step the environment
        if not done:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        # Render the observation to the human-visible screen
        # Need to transpose back from (H, W, C) to (W, H, C) for pygame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        human_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Control the frame rate
        env.clock.tick(60)
        
    env.close()