
# Generated: 2025-08-27T20:27:07.182684
# Source Brief: brief_02462.md
# Brief Index: 2462

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
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
        "Controls: Use ←→ to aim your jump. Press SPACE to jump. Hold SHIFT while jumping for a higher leap."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Guide a space hopper upwards by leaping between procedurally generated platforms, "
        "balancing risk and reward to reach the top in the shortest time."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen dimensions
        self.WIDTH, self.HEIGHT = 640, 400

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
        self.font = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 40, bold=True)

        # Colors
        self.COLOR_BG_TOP = (10, 0, 20)
        self.COLOR_BG_BOTTOM = (0, 0, 0)
        self.COLOR_PLAYER = (0, 255, 128)
        self.COLOR_PLAYER_GLOW = (128, 255, 200)
        self.COLOR_PLATFORM = (255, 255, 255)
        self.COLOR_PARTICLE = (220, 220, 220)
        self.COLOR_TEXT = (255, 255, 255)

        # Game constants
        self.GRAVITY = 0.4
        self.MAX_STEPS = 1000
        self.WIN_LEVEL = 10

        # Pre-render background
        self.background = pygame.Surface((self.WIDTH, self.HEIGHT))
        for y in range(self.HEIGHT):
            interp = y / self.HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(self.background, color, (0, y), (self.WIDTH, y))

        # State variables are initialized in reset()
        self.hopper = {}
        self.platforms = []
        self.particles = []
        self.camera_y = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.game_won = False
        self.successful_jumps = 0
        self.gap_scale = 1.0
        self.highest_platform_y = 0
        self.last_y_pos = 0
        self.last_space_held = False
        self.last_platform_y = 0
        self.np_random = None
        
        # Initialize state variables
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.successful_jumps = 0
        self.gap_scale = 1.0
        self.camera_y = 0
        self.particles.clear()
        
        # Player state
        self.hopper = {
            "pos": pygame.Vector2(self.WIDTH / 2, self.HEIGHT - 50),
            "vel": pygame.Vector2(0, 0),
            "size": 20,
            "on_platform": True
        }
        
        # Platform generation
        self.platforms.clear()
        initial_platform = pygame.Rect(self.WIDTH/2 - 50, self.HEIGHT - 30, 100, 15)
        self.platforms.append(initial_platform)
        self.highest_platform_y = initial_platform.y
        self.last_platform_y = initial_platform.y
        self._generate_initial_platforms()

        self.last_y_pos = self.hopper['pos'].y
        self.last_space_held = False

        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def _generate_initial_platforms(self):
        # Generate enough platforms to fill the screen initially
        while self.highest_platform_y > -self.HEIGHT:
            self._generate_next_platform()

    def _generate_next_platform(self):
        last_platform = self.platforms[-1]
        
        base_gap_y = 100
        gap_y = self.np_random.uniform(base_gap_y * 0.8, base_gap_y * 1.2) * self.gap_scale
        new_y = last_platform.y - gap_y
        
        min_width, max_width = 60, 150
        width = self.np_random.uniform(min_width, max_width)
        
        max_dx = 180
        new_x = self.np_random.uniform(
            max(20, last_platform.centerx - max_dx),
            min(self.WIDTH - 20 - width, last_platform.centerx + max_dx)
        )
        
        new_platform = pygame.Rect(new_x, new_y, width, 15)
        self.platforms.append(new_platform)
        self.highest_platform_y = new_y
    
    def step(self, action):
        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        shift_held = action[2] == 1
        
        reward = 0
        
        self._handle_input(movement, space_held, shift_held)
        
        # Store pre-update position for reward calculation
        self.last_y_pos = self.hopper['pos'].y
        
        self._update_hopper_physics()
        
        # Reward for upward movement
        reward += (self.last_y_pos - self.hopper['pos'].y) * 0.1
        
        # Small penalty for horizontal velocity
        reward -= abs(self.hopper['vel'].x) * 0.005
        
        landing_reward = self._check_collisions()
        reward += landing_reward

        self._update_camera()
        self._update_platforms()
        self._update_particles()
        
        # Update game logic
        self.steps += 1
        terminated = self._check_termination()

        if terminated:
            if self.game_won:
                reward = 100.0
            elif self.game_over:
                reward = -100.0
        
        self.score += reward

        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_input(self, movement, space_held, shift_held):
        space_just_pressed = space_held and not self.last_space_held
        self.last_space_held = space_held

        if self.hopper['on_platform'] and space_just_pressed:
            # Sound: Jump
            self.hopper['on_platform'] = False
            
            aim_x = 0
            if movement == 3:  # Left
                aim_x = -1
            elif movement == 4: # Right
                aim_x = 1
            
            jump_power_y = -10 if shift_held else -8
            jump_power_x = 5
            
            self.hopper['vel'].y = jump_power_y
            self.hopper['vel'].x = aim_x * jump_power_x

    def _update_hopper_physics(self):
        if not self.hopper['on_platform']:
            self.hopper['vel'].y += self.GRAVITY
        
        self.hopper['pos'] += self.hopper['vel']
        
        # Wall bouncing
        if self.hopper['pos'].x < 0 or self.hopper['pos'].x + self.hopper['size'] > self.WIDTH:
            self.hopper['vel'].x *= -0.8
            self.hopper['pos'].x = max(0, min(self.hopper['pos'].x, self.WIDTH - self.hopper['size']))

    def _check_collisions(self):
        reward = 0
        hopper_rect = pygame.Rect(self.hopper['pos'].x, self.hopper['pos'].y, self.hopper['size'], self.hopper['size'])
        
        if self.hopper['vel'].y > 0: # Only check for landing when falling
            for platform in self.platforms:
                if hopper_rect.colliderect(platform) and hopper_rect.bottom < platform.bottom:
                    prev_bottom = self.hopper['pos'].y + self.hopper['size'] - self.hopper['vel'].y
                    if prev_bottom <= platform.top:
                        self.hopper['on_platform'] = True
                        self.hopper['vel'] = pygame.Vector2(0, 0)
                        self.hopper['pos'].y = platform.top - self.hopper['size']
                        
                        self._create_landing_particles(hopper_rect.midbottom)
                        # Sound: Land
                        
                        reward += 1.0 # Base landing reward
                        self.successful_jumps += 1
                        
                        # Risky jump reward
                        jump_gap = self.last_platform_y - platform.y
                        avg_gap = 100 * self.gap_scale
                        if jump_gap > avg_gap * 1.1:
                            reward += 2.0 # Risky jump bonus
                        elif jump_gap < avg_gap * 0.9:
                            reward -= 0.2 # Safe jump penalty
                            
                        self.last_platform_y = platform.y

                        if self.successful_jumps > 0 and self.successful_jumps % 5 == 0:
                            self.gap_scale *= 1.05
                        
                        return reward
        return reward

    def _update_camera(self):
        scroll_threshold = self.HEIGHT / 3
        if self.hopper['pos'].y < self.camera_y + scroll_threshold:
            self.camera_y = self.hopper['pos'].y - scroll_threshold
    
    def _update_platforms(self):
        if not self.platforms or self.platforms[-1].y - self.camera_y > -50:
             self._generate_next_platform()
        
        self.platforms = [p for p in self.platforms if p.bottom > self.camera_y]

    def _create_landing_particles(self, pos):
        for _ in range(15):
            vel = pygame.Vector2(self.np_random.uniform(-2, 2), self.np_random.uniform(-3, -0.5))
            self.particles.append({
                "pos": pygame.Vector2(pos),
                "vel": vel,
                "life": self.np_random.integers(15, 30),
                "radius": self.np_random.uniform(1, 3)
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'].y += 0.1 # Particle gravity
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _check_termination(self):
        if self.hopper['pos'].y > self.camera_y + self.HEIGHT + 50:
            self.game_over = True
            return True
        if self.successful_jumps >= self.WIN_LEVEL:
            self.game_won = True
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False
    
    def _get_observation(self):
        # Clear screen with background
        self.screen.blit(self.background, (0, 0))
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render platforms
        for p in self.platforms:
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, p.move(0, -self.camera_y))
        
        # Render particles
        for p in self.particles:
            pos = (int(p['pos'].x), int(p['pos'].y - self.camera_y))
            pygame.draw.circle(self.screen, self.COLOR_PARTICLE, pos, int(p['radius']))
            
        # Render hopper
        hopper_rect = pygame.Rect(
            self.hopper['pos'].x, self.hopper['pos'].y - self.camera_y,
            self.hopper['size'], self.hopper['size']
        )
        
        # Glow effect
        glow_size = int(self.hopper['size'] * 1.8)
        glow_surf = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*self.COLOR_PLAYER_GLOW, 50), (glow_size // 2, glow_size // 2), glow_size // 2)
        self.screen.blit(glow_surf, (hopper_rect.centerx - glow_size // 2, hopper_rect.centery - glow_size // 2))
        
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, hopper_rect)

    def _render_ui(self):
        score_text = self.font.render(f"SCORE: {self.score:.0f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        level_text = self.font.render(f"LEVEL: {self.successful_jumps}/{self.WIN_LEVEL}", True, self.COLOR_TEXT)
        self.screen.blit(level_text, (10, 35))
        
        steps_text = self.font.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (self.WIDTH - steps_text.get_width() - 10, 10))

        if self.game_over or self.game_won:
            message = "YOU WIN!" if self.game_won else "GAME OVER"
            msg_surf = self.font_large.render(message, True, self.COLOR_TEXT)
            msg_rect = msg_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(msg_surf, msg_rect)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.successful_jumps,
            "game_over": self.game_over,
            "game_won": self.game_won
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
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

if __name__ == '__main__':
    # This block allows you to run the file directly to test the environment
    env = GameEnv()
    obs, info = env.reset()
    
    # Pygame setup for human play
    render_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Space Hopper")
    
    terminated = False
    
    # Game loop
    while not terminated:
        # Default action is no-op
        movement = 0
        space = 0
        shift = 0

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        # Key presses
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space = 1
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift = 1

        action = [movement, space, shift]
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Cap the frame rate
        env.clock.tick(30)

        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Level: {info['level']}")
            pygame.time.wait(2000)
            
    env.close()