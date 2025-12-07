import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = "Controls: ←→ to move, ↑ to jump."

    # Must be a short, user-facing description of the game:
    game_description = (
        "A minimalist platformer where an agent navigates jumping between platforms and collecting gems to reach the goal."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 1000

    # Colors
    COLOR_BG = (25, 28, 36)
    COLOR_PLAYER = (255, 87, 87)
    COLOR_PLAYER_GLOW = (255, 87, 87, 50)
    COLOR_PLATFORM = (60, 65, 80)
    COLOR_GEM = (255, 220, 70)
    COLOR_GEM_GLOW = (255, 220, 70, 70)
    COLOR_GOAL = (87, 255, 150)
    COLOR_GOAL_GLOW = (87, 255, 150, 70)
    COLOR_TEXT = (240, 240, 240)
    
    # Physics
    GRAVITY = 0.8
    JUMP_STRENGTH = -15
    MOVE_SPEED = 1.0
    MAX_VEL_X = 6
    FRICTION = 0.85
    PLAYER_SIZE = 20

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Set up headless Pygame
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.init()
        pygame.font.init()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 24, bold=True)
        
        # Initialize state variables
        self.player_pos = None
        self.player_vel = None
        self.player_on_ground = False
        self.platforms = []
        self.gems = []
        self.goal = None
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.np_random = None

        # self.validate_implementation() # Optional: can be removed
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.particles = []
        
        self._generate_level()
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def _generate_level(self):
        # Start with a safe, screen-wide base platform to ensure stability on no-op.
        start_plat_y = 50
        self.platforms = [pygame.Rect(0, start_plat_y, self.WIDTH, 20)]
        self.player_pos = pygame.Vector2(self.WIDTH // 2, self.platforms[0].top - self.PLAYER_SIZE)
        self.player_vel = pygame.Vector2(0, 0)
        self.player_on_ground = True
        
        self.gems = []
        
        # Generate subsequent platforms
        last_platform = self.platforms[0]
        for i in range(10):
            px = last_platform.centerx
            py = last_platform.top
            
            dx = self.np_random.integers(-150, 151)
            dy = self.np_random.integers(60, 101)
            
            # For the first platform after the wide start, base its x on center screen
            if i == 0:
                px = self.WIDTH // 2

            new_x = np.clip(px + dx, 50, self.WIDTH - 150)
            new_y = np.clip(py + dy, last_platform.bottom + 50, self.HEIGHT - 50)
            
            width = self.np_random.integers(80, 151)
            
            new_platform = pygame.Rect(new_x - width // 2, new_y, width, 20)
            self.platforms.append(new_platform)
            
            # Add a gem above the new platform
            if self.np_random.random() < 0.8:
                gem_x = new_platform.centerx
                gem_y = new_platform.top - 30
                self.gems.append(pygame.Rect(gem_x - 5, gem_y - 5, 10, 10))

            last_platform = new_platform
            
        # Set the last platform as the goal
        self.goal = self.platforms.pop()
        
    def step(self, action):
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        # space_held = action[1] == 1 - no effect
        # shift_held = action[2] == 1 - no effect
        
        reward = -0.02 # Time penalty
        
        # --- Player Logic ---
        # Horizontal movement
        is_moving_horizontally = False
        if movement == 3:  # Left
            self.player_vel.x -= self.MOVE_SPEED
            is_moving_horizontally = True
        elif movement == 4:  # Right
            self.player_vel.x += self.MOVE_SPEED
            is_moving_horizontally = True
            
        # Jumping
        if movement == 1 and self.player_on_ground:
            self.player_vel.y = self.JUMP_STRENGTH
            self.player_on_ground = False
            # sfx: jump
            # Reward for risky jumps
            if self.player_pos.y < self.HEIGHT * 0.25:
                 reward += 0.1

        # Penalty for safe moves
        if is_moving_horizontally and self.player_on_ground and self.player_pos.y > self.HEIGHT * 0.5:
            reward -= 0.2

        # Apply physics
        self.player_vel.x = np.clip(self.player_vel.x, -self.MAX_VEL_X, self.MAX_VEL_X)
        if not is_moving_horizontally and self.player_on_ground:
            self.player_vel.x *= self.FRICTION
        
        self.player_vel.y += self.GRAVITY
        
        # Update position
        self.player_pos += self.player_vel
        
        player_rect = pygame.Rect(self.player_pos.x, self.player_pos.y, self.PLAYER_SIZE, self.PLAYER_SIZE)

        # --- Collision Detection ---
        self.player_on_ground = False
        for plat in self.platforms:
            if player_rect.colliderect(plat) and self.player_vel.y > 0:
                # Check if player was above the platform in the previous frame
                prev_player_bottom = self.player_pos.y + self.PLAYER_SIZE - self.player_vel.y
                if prev_player_bottom <= plat.top + 1: # Add tolerance
                    self.player_pos.y = plat.top - self.PLAYER_SIZE
                    self.player_vel.y = 0
                    self.player_on_ground = True
                    break
        
        # Screen boundaries
        if self.player_pos.x < 0:
            self.player_pos.x = 0
            self.player_vel.x = 0
        if self.player_pos.x > self.WIDTH - self.PLAYER_SIZE:
            self.player_pos.x = self.WIDTH - self.PLAYER_SIZE
            self.player_vel.x = 0

        # --- Collectibles & Goal ---
        player_rect.topleft = self.player_pos
        
        # Gem collection
        collected_indices = player_rect.collidelistall(self.gems)
        if collected_indices:
            for i in sorted(collected_indices, reverse=True):
                gem_rect = self.gems.pop(i)
                self.score += 1
                reward += 1.0
                self._create_particles(gem_rect.center, self.COLOR_GEM, 15)
        
        # --- Termination Conditions ---
        terminated = False
        truncated = False
        
        # Goal reached
        if player_rect.colliderect(self.goal):
            reward += 100.0
            self.score += 10
            terminated = True
            self.game_over = True
            self._create_particles(player_rect.center, self.COLOR_GOAL, 50)
        
        # Fell off screen
        if self.player_pos.y > self.HEIGHT:
            reward = -100.0
            terminated = True
            self.game_over = True
            
        # Max steps
        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            truncated = True # Use truncated for time limit
            self.game_over = True

        # Update particles
        self._update_particles()
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )
    
    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 3 + 1
            vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            lifespan = self.np_random.integers(15, 31)
            self.particles.append({"pos": pygame.Vector2(pos), "vel": vel, "lifespan": lifespan, "color": color})

    def _update_particles(self):
        for p in self.particles:
            p["pos"] += p["vel"]
            p["lifespan"] -= 1
        self.particles = [p for p in self.particles if p["lifespan"] > 0]

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Platforms
        for plat in self.platforms:
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, plat, border_radius=3)

        # Goal
        pygame.gfxdraw.box(self.screen, self.goal, self.COLOR_GOAL_GLOW)
        pygame.draw.rect(self.screen, self.COLOR_GOAL, self.goal, border_radius=3)

        # Gems
        for gem in self.gems:
            pygame.gfxdraw.filled_circle(self.screen, gem.centerx, gem.centery, 8, self.COLOR_GEM_GLOW)
            pygame.gfxdraw.filled_circle(self.screen, gem.centerx, gem.centery, 6, self.COLOR_GEM)
            pygame.gfxdraw.aacircle(self.screen, gem.centerx, gem.centery, 6, self.COLOR_GEM)
        
        # Particles
        for p in self.particles:
            alpha = int(255 * (p["lifespan"] / 30.0))
            # Create a temporary surface for the particle to handle alpha
            particle_surf = pygame.Surface((p["lifespan"], p["lifespan"]), pygame.SRCALPHA)
            size = int(max(1, 3 * (p["lifespan"] / 30.0)))
            pygame.draw.circle(particle_surf, (*p["color"], alpha), (size, size), size)
            self.screen.blit(particle_surf, (int(p["pos"].x) - size, int(p["pos"].y) - size), special_flags=pygame.BLEND_RGBA_ADD)

        # Player
        player_rect = pygame.Rect(
            int(self.player_pos.x), int(self.player_pos.y), 
            self.PLAYER_SIZE, self.PLAYER_SIZE
        )
        glow_size = int(self.PLAYER_SIZE * 1.5)
        glow_rect = pygame.Rect(0,0,glow_size,glow_size)
        glow_rect.center = player_rect.center
        pygame.gfxdraw.box(self.screen, glow_rect, self.COLOR_PLAYER_GLOW)
        
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=3)
    
    def _render_ui(self):
        score_text = self.font.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        steps_text = self.font.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (self.WIDTH - steps_text.get_width() - 10, 10))
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        print("Validating implementation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert obs.dtype == np.uint8
        assert isinstance(info, dict)
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv()
    try:
        env.validate_implementation()
        obs, info = env.reset(seed=42)
        print("Initial observation shape:", obs.shape)
        print("Initial info:", info)

        terminated = False
        truncated = False
        total_reward = 0
        
        # Run for a few steps with random actions
        for i in range(200):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                print(f"Episode finished after {i+1} steps.")
                break
                
        print("Final observation shape:", obs.shape)
        print("Final info:", info)
        print(f"Total reward over {i+1} steps:", total_reward)
        
    finally:
        env.close()

    # To visualize the game, you would need to remove the dummy video driver
    # and add a render loop.
    print("\nTo visualize, uncomment the rendering code in the main block.")
    #
    # import sys
    # os.environ.pop("SDL_VIDEODRIVER", None)
    # env = GameEnv(render_mode="rgb_array")
    # obs, info = env.reset()
    # screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    # pygame.display.set_caption("Platformer")
    # clock = pygame.time.Clock()
    #
    # running = True
    # while running:
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             running = False
    #         if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
    #             running = False
    #
    #     # Simple keyboard control for testing
    #     keys = pygame.key.get_pressed()
    #     move = 0
    #     if keys[pygame.K_UP]: move = 1
    #     if keys[pygame.K_LEFT]: move = 3
    #     if keys[pygame.K_RIGHT]: move = 4
    #     action = [move, 0, 0] # No space or shift
    #
    #     obs, reward, terminated, truncated, info = env.step(action)
    #
    #     if terminated or truncated:
    #         print(f"Game Over! Score: {info['score']}")
    #         obs, info = env.reset()
    #
    #     # Render to the display
    #     surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
    #     screen.blit(surf, (0, 0))
    #     pygame.display.flip()
    #     clock.tick(GameEnv.FPS)
    #
    # env.close()
    # pygame.quit()
    # sys.exit()