
# Generated: 2025-08-27T21:18:51.849294
# Source Brief: brief_02748.md
# Brief Index: 2748

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to aim jump. Space for a high jump, Shift for a low jump. "
        "Combine aiming and jumping."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A minimalist platformer. Jump between procedurally generated platforms to collect all the stars. "
        "Falling off the screen ends the game."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    MAX_STEPS = 1000
    NUM_STARS = 25
    GRAVITY = 0.4
    NINJA_SIZE = 20
    STAR_RADIUS = 8
    PARTICLE_LIFESPAN = 20

    # Jump strengths
    HIGH_JUMP_POWER = -10
    LOW_JUMP_POWER = -7
    HORIZONTAL_JUMP_POWER = 6
    RISKY_JUMP_THRESHOLD = 9

    # Colors
    COLOR_BG = (15, 15, 15)  # Near-black
    COLOR_NINJA = (255, 50, 50)
    COLOR_PLATFORM = (240, 240, 240)
    COLOR_STAR = (255, 220, 0)
    COLOR_PARTICLE = (200, 200, 200)
    COLOR_TEXT = (255, 255, 255)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

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
        self.font = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        # Game state variables are initialized in reset()
        self.ninja_pos = None
        self.ninja_vel = None
        self.platforms = []
        self.stars = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.on_ground = False

        # Initialize state variables
        self.reset()
        
        # Run validation check
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.particles = []

        self._generate_level()
        
        # Place ninja on the first platform
        start_platform = self.platforms[0]
        self.ninja_pos = pygame.Vector2(
            start_platform.centerx, start_platform.top - self.NINJA_SIZE
        )
        self.ninja_vel = pygame.Vector2(0, 0)
        self.on_ground = True

        return self._get_observation(), self._get_info()

    def _generate_level(self):
        self.platforms = []
        self.stars = []

        # Create starting platform
        plat_w = 120
        plat_h = 20
        start_platform = pygame.Rect(
            self.WIDTH / 2 - plat_w / 2, self.HEIGHT - 40, plat_w, plat_h
        )
        self.platforms.append(start_platform)
        
        last_platform = start_platform
        
        for i in range(self.NUM_STARS):
            # Place star on the platform
            star_pos = pygame.Vector2(last_platform.centerx, last_platform.top - self.STAR_RADIUS - 5)
            self.stars.append(star_pos)
            
            if len(self.platforms) >= self.NUM_STARS:
                break
            
            # Generate next platform
            plat_w = self.np_random.integers(60, 110)
            plat_h = 20
            
            # Ensure next platform is reachable
            max_h_jump = abs(self.HIGH_JUMP_POWER)
            max_reach_y = (max_h_jump ** 2) / (2 * self.GRAVITY) # Kinematic equation: vf^2 = vi^2 + 2ad
            
            dx = self.np_random.uniform(-150, 150)
            dy = self.np_random.uniform(max_reach_y * 0.3, max_reach_y * 0.85)

            new_x = last_platform.centerx + dx
            new_y = last_platform.top - dy
            
            # Clamp to screen bounds
            new_x = np.clip(new_x, plat_w / 2, self.WIDTH - plat_w / 2)
            new_y = np.clip(new_y, 40, self.HEIGHT - 80)

            new_platform = pygame.Rect(new_x - plat_w / 2, new_y, plat_w, plat_h)
            self.platforms.append(new_platform)
            last_platform = new_platform
        
        # The first star is on the start platform, so we remove the extra one
        self.stars.pop(0)


    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0.1 # Small reward for surviving a step

        # --- 1. Handle Input & Jumping ---
        movement = action[0]
        space_held = action[1] == 1
        shift_held = action[2] == 1
        
        jump_intent = space_held or shift_held or (movement in [1, 3, 4])
        
        if self.on_ground and jump_intent:
            # Determine jump power
            jump_power = 0
            if space_held:
                jump_power = self.HIGH_JUMP_POWER
            elif shift_held:
                jump_power = self.LOW_JUMP_POWER
            
            # Apply jump velocity
            self.ninja_vel.y = jump_power
            
            # Apply directional movement
            if movement == 1: # Up
                self.ninja_vel.y += self.LOW_JUMP_POWER / 2
            elif movement == 3: # Left
                self.ninja_vel.x = -self.HORIZONTAL_JUMP_POWER
            elif movement == 4: # Right
                self.ninja_vel.x = self.HORIZONTAL_JUMP_POWER
            
            # If only aiming left/right without a vertical jump, give a small hop
            if movement in [3, 4] and not (space_held or shift_held):
                self.ninja_vel.y = self.LOW_JUMP_POWER / 1.5

            self.on_ground = False
            
            # Calculate risky/safe jump reward
            jump_magnitude = self.ninja_vel.magnitude()
            if jump_magnitude > self.RISKY_JUMP_THRESHOLD:
                reward += 2.0 # Risky jump
            else:
                reward -= 1.0 # Safe jump
        
        # --- 2. Update Physics ---
        self.ninja_vel.y += self.GRAVITY
        
        # Air friction/drag
        self.ninja_vel.x *= 0.95
        if abs(self.ninja_vel.x) < 0.1:
            self.ninja_vel.x = 0

        # Update position
        self.ninja_pos += self.ninja_vel
        ninja_rect = pygame.Rect(self.ninja_pos.x, self.ninja_pos.y, self.NINJA_SIZE, self.NINJA_SIZE)

        # --- 3. Handle Collisions ---
        self.on_ground = False
        for plat in self.platforms:
            if ninja_rect.colliderect(plat) and self.ninja_vel.y > 0:
                # Check if the ninja was above the platform in the previous frame
                if (self.ninja_pos.y + self.NINJA_SIZE - self.ninja_vel.y) <= plat.top:
                    self.on_ground = True
                    self.ninja_pos.y = plat.top - self.NINJA_SIZE
                    self.ninja_vel.y = 0
                    self.ninja_vel.x = 0 # Stop horizontal movement on landing
                    self._create_particles(pygame.Vector2(ninja_rect.midbottom), 20)
                    # sfx: landing_sound
                    break
        
        # --- 4. Collect Stars ---
        collected_indices = []
        for i, star_pos in enumerate(self.stars):
            if ninja_rect.collidepoint(star_pos):
                collected_indices.append(i)
                self.score += 1
                reward += 10.0
                self._create_particles(star_pos, 30, self.COLOR_STAR)
                # sfx: star_collect_sound

        # Remove collected stars safely
        for i in sorted(collected_indices, reverse=True):
            del self.stars[i]
            
        # --- 5. Update Particles ---
        self._update_particles()

        # --- 6. Check Termination Conditions ---
        self.steps += 1
        terminated = False
        
        # Fell off screen
        if self.ninja_pos.y > self.HEIGHT:
            terminated = True
            self.game_over = True
        
        # Collected all stars
        if not self.stars:
            terminated = True
            self.game_over = True
            reward += 100.0 # Victory bonus
            
        # Max steps reached
        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _create_particles(self, position, count, color=COLOR_PARTICLE):
        for _ in range(count):
            vel = pygame.Vector2(self.np_random.uniform(-2, 2), self.np_random.uniform(-3, 0))
            self.particles.append({
                'pos': position.copy(),
                'vel': vel,
                'life': self.np_random.integers(10, self.PARTICLE_LIFESPAN),
                'max_life': self.PARTICLE_LIFESPAN,
                'color': color
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render platforms
        for plat in self.platforms:
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, plat)

        # Render stars
        for star_pos in self.stars:
            x, y = int(star_pos.x), int(star_pos.y)
            pygame.gfxdraw.filled_circle(self.screen, x, y, self.STAR_RADIUS, self.COLOR_STAR)
            pygame.gfxdraw.aacircle(self.screen, x, y, self.STAR_RADIUS, self.COLOR_STAR)

        # Render particles
        for p in self.particles:
            pos = p['pos']
            life_ratio = p['life'] / p['max_life']
            radius = int(life_ratio * 4)
            if radius > 0:
                alpha = int(life_ratio * 255)
                color = p['color']
                # Create a temporary surface for alpha blending
                temp_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, (*color, alpha), (radius, radius), radius)
                self.screen.blit(temp_surf, (int(pos.x - radius), int(pos.y - radius)))

        # Render ninja
        ninja_rect = pygame.Rect(self.ninja_pos.x, self.ninja_pos.y, self.NINJA_SIZE, self.NINJA_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_NINJA, ninja_rect)

    def _render_ui(self):
        score_text = self.font.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        stars_text = self.font.render(f"Stars: {len(self.stars)}/{self.NUM_STARS}", True, self.COLOR_TEXT)
        self.screen.blit(stars_text, (self.WIDTH - stars_text.get_width() - 10, 10))
        
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            end_text_str = "VICTORY!" if not self.stars else "GAME OVER"
            end_text = self.font.render(end_text_str, True, self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 - 20))
            self.screen.blit(end_text, text_rect)
            
            final_score_text = self.font_small.render(f"Final Score: {self.score}", True, self.COLOR_TEXT)
            score_rect = final_score_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 + 20))
            self.screen.blit(final_score_text, score_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stars_remaining": len(self.stars),
            "ninja_pos": (self.ninja_pos.x, self.ninja_pos.y),
        }
        
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
if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    
    # --- To play manually ---
    # This requires setting up a pygame window to display the frames
    # and to capture keyboard events.
    
    # Set render_mode to "human" if you add a human render mode
    # For this example, we'll just show the rgb_array in a window
    
    window = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Ninja Platformer")
    
    obs, info = env.reset()
    terminated = False
    
    # Key mapping
    key_map = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }
    
    print(GameEnv.game_description)
    print(GameEnv.user_guide)

    while not terminated:
        # Default action is a no-op
        movement = 0
        space = 0
        shift = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]:
            space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift = 1
            
        for key, move_val in key_map.items():
            if keys[key]:
                movement = move_val
                break # Prioritize one movement key

        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        window.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Since auto_advance is False, we control the step rate here
        env.clock.tick(30) # Limit to 30 steps per second for playability

    env.close()
    pygame.quit()