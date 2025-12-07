import os
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import os
import pygame



class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ↑ to jump. Use ←→ to move in the air. Hold Space to fast-fall."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Leap between floating platforms to reach the sky! Master your air control to land on "
        "ever-higher platforms in this fast-paced arcade challenge."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 2500
        self.NUM_PLATFORMS = 50
        self.INITIAL_LIVES = 3

        # Physics constants
        self.GRAVITY = 0.8
        self.JUMP_STRENGTH = 15
        self.AIR_CONTROL_FORCE = 1.2
        self.MAX_VX = 7
        self.AIR_DRAG = 0.92
        self.FAST_FALL_SPEED = 12

        # Color Palette
        self.COLOR_BG_TOP = (10, 20, 40)
        self.COLOR_BG_BOTTOM = (0, 0, 10)
        self.COLOR_PLAYER = (255, 255, 0) # Bright Yellow
        self.COLOR_PLAYER_OUTLINE = (200, 200, 0)
        self.COLOR_PLATFORM_START = (0, 100, 200) # Blue
        self.COLOR_PLATFORM_END = (150, 220, 255) # Light Cyan
        self.COLOR_PLATFORM_EDGE = (255, 255, 255)
        self.COLOR_PARTICLE = (220, 220, 220)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_TEXT_SHADOW = (50, 50, 50)
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        # Etc...        
        
        # Initialize state variables
        # self.reset() # reset is called by the test harness, no need to call it here
        # self.validate_implementation() # Optional validation call

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.lives = self.INITIAL_LIVES
        self.game_over = False
        self.particles = []
        self.highest_platform_idx = 0

        self._generate_platforms()
        
        start_platform = self.platforms[0]
        self.player_pos = np.array([start_platform.centerx, start_platform.top - 15], dtype=np.float32)
        self.player_vel = np.zeros(2, dtype=np.float32)
        self.on_ground = True
        
        self.camera_y = 0

        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        shift_held = action[2] == 1  # Boolean (unused)
        
        # Update game logic
        reward = self._update_game_logic(movement, space_held)
        
        self.steps += 1
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        
        if terminated and not self.game_over: # Max steps reached
             # Small penalty for running out of time
             reward -= 10

        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _update_game_logic(self, movement, space_held):
        reward = 0
        
        # 1. Apply player input
        if self.on_ground and movement == 1: # Up (Jump)
            self.player_vel[1] = -self.JUMP_STRENGTH
            self.on_ground = False
            # sfx: jump_sound()
        
        if not self.on_ground:
            if movement == 3: # Left
                self.player_vel[0] = max(-self.MAX_VX, self.player_vel[0] - self.AIR_CONTROL_FORCE)
            elif movement == 4: # Right
                self.player_vel[0] = min(self.MAX_VX, self.player_vel[0] + self.AIR_CONTROL_FORCE)
            
            if space_held and self.player_vel[1] > 0: # Fast-fall
                self.player_vel[1] = max(self.player_vel[1], self.FAST_FALL_SPEED)

        # 2. Apply physics
        self.player_vel[1] += self.GRAVITY
        self.player_vel[0] *= self.AIR_DRAG
        
        self.player_pos += self.player_vel
        
        # Continuous reward for vertical movement
        reward -= self.player_vel[1] * 0.01 # Reward for going up, penalize for going down

        # 3. Collision and boundary checks
        self._handle_collisions_and_boundaries()
        
        # 4. Check for platform landing reward
        if self.on_ground:
            landed_on_new = False
            for i, plat in enumerate(self.platforms):
                # check if player feet are on this platform
                player_feet_rect = pygame.Rect(int(self.player_pos[0]) - 10, int(self.player_pos[1]) + 9, 20, 2)
                if player_feet_rect.colliderect(plat):
                    if i > self.highest_platform_idx:
                        self.highest_platform_idx = i
                        landed_on_new = True
                        break
            if landed_on_new:
                # sfx: new_platform_ding()
                new_platform_reward = 1.0 + (self.highest_platform_idx / self.NUM_PLATFORMS) * 4 # Scale reward with height
                reward += new_platform_reward
                self.score += int(new_platform_reward * 10)

        # 5. Update camera and particles
        self._update_camera()
        self._update_particles()
        
        # 6. Check for win condition
        if self.highest_platform_idx >= self.NUM_PLATFORMS - 1:
            reward += 100
            self.score += 1000
            self.game_over = True
        
        # Check for life loss
        if self.player_pos[1] > self.camera_y + self.HEIGHT + 50:
            self.lives -= 1
            reward -= 5
            # sfx: life_lost_sound()
            if self.lives <= 0:
                self.game_over = True
                reward -= 10 # Extra penalty for game over
            else:
                self._reset_player_position()

        return reward

    def _handle_collisions_and_boundaries(self):
        # Horizontal screen boundaries
        if self.player_pos[0] < 10:
            self.player_pos[0] = 10
            self.player_vel[0] = 0
        elif self.player_pos[0] > self.WIDTH - 10:
            self.player_pos[0] = self.WIDTH - 10
            self.player_vel[0] = 0

        # Platform collisions
        self.on_ground = False
        if self.player_vel[1] > 0: # Only check for landing if falling
            player_rect = pygame.Rect(int(self.player_pos[0]) - 10, int(self.player_pos[1]) - 10, 20, 20)
            for plat in self.platforms:
                # Simple AABB check for landing on top of a platform
                if (player_rect.right > plat.left and player_rect.left < plat.right and
                    player_rect.bottom <= plat.top and player_rect.bottom + self.player_vel[1] >= plat.top):
                    
                    self.player_pos[1] = plat.top - 10
                    self.player_vel[1] = 0
                    self.on_ground = True
                    self._create_landing_particles(self.player_pos[0], plat.top)
                    # sfx: land_thud()
                    break
    
    def _reset_player_position(self):
        reset_platform = self.platforms[self.highest_platform_idx]
        self.player_pos = np.array([reset_platform.centerx, reset_platform.top - 15], dtype=np.float32)
        self.player_vel = np.zeros(2, dtype=np.float32)
        self.on_ground = True

    def _generate_platforms(self):
        self.platforms = []
        # First platform
        start_plat = pygame.Rect(self.WIDTH // 2 - 60, self.HEIGHT - 40, 120, 15)
        self.platforms.append(start_plat)
        
        last_y = start_plat.y
        last_x = start_plat.centerx
        
        for i in range(1, self.NUM_PLATFORMS):
            max_h_dist = 80 + i * 2.5  # Horizontal distance variance increases
            min_v_dist = 60
            max_v_dist = 120 + i * 1.0 # Vertical distance increases
            
            dy = self.np_random.uniform(min_v_dist, max_v_dist)
            dx = self.np_random.uniform(-max_h_dist, max_h_dist)
            
            new_y = last_y - dy
            new_x = last_x + dx
            
            width = max(40, 100 - i * 1.2)
            
            # Ensure platforms don't go off-screen
            new_x = np.clip(new_x, width / 2 + 10, self.WIDTH - width / 2 - 10)
            
            plat = pygame.Rect(new_x - width / 2, new_y, width, 15)
            self.platforms.append(plat)
            
            last_y = new_y
            last_x = new_x

    def _create_landing_particles(self, x, y):
        for _ in range(10):
            vel_x = self.np_random.uniform(-2, 2)
            vel_y = self.np_random.uniform(-3, -0.5)
            life = self.np_random.integers(10, 20)
            self.particles.append({"pos": [x, y], "vel": [vel_x, vel_y], "life": life, "max_life": life})

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][1] += 0.2 # Particles have slight gravity
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

    def _update_camera(self):
        # Smoothly follow the player vertically
        target_camera_y = self.player_pos[1] - self.HEIGHT * 0.5
        self.camera_y += (target_camera_y - self.camera_y) * 0.08

    def _get_observation(self):
        # Clear screen with background
        self._render_background()
        
        # Render all game elements
        self._render_game_elements()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        # Draw a vertical gradient
        for y in range(self.HEIGHT):
            progress = y / self.HEIGHT
            color = [
                int(self.COLOR_BG_TOP[i] * (1 - progress) + self.COLOR_BG_BOTTOM[i] * progress)
                for i in range(3)
            ]
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

    def _render_game_elements(self):
        # Draw platforms
        for i, plat in enumerate(self.platforms):
            cam_x, cam_y = int(plat.x), int(plat.y - self.camera_y)
            if cam_y > self.HEIGHT or cam_y + plat.height < 0:
                continue

            progress = min(1.0, i / (self.NUM_PLATFORMS - 1)) if self.NUM_PLATFORMS > 1 else 0
            body_color = [
                int(self.COLOR_PLATFORM_START[c] * (1 - progress) + self.COLOR_PLATFORM_END[c] * progress)
                for c in range(3)
            ]
            
            pygame.draw.rect(self.screen, body_color, (cam_x, cam_y, plat.width, plat.height))
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM_EDGE, (cam_x, cam_y, plat.width, plat.height), 2)

        # Draw particles
        for p in self.particles:
            cam_x, cam_y = int(p["pos"][0]), int(p["pos"][1] - self.camera_y)
            radius = int(max(0, (p["life"] / p["max_life"]) * 4))
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, cam_x, cam_y, radius, self.COLOR_PARTICLE)

        # Draw player
        player_cam_x = int(self.player_pos[0])
        player_cam_y = int(self.player_pos[1] - self.camera_y)
        player_rect = pygame.Rect(player_cam_x - 10, player_cam_y - 10, 20, 20)
        
        pygame.draw.rect(self.screen, self.COLOR_PLAYER_OUTLINE, player_rect.inflate(4,4), border_radius=3)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=3)

    def _render_ui(self):
        # Helper to render text with shadow
        def draw_text(text, font, color, x, y, align="topleft"):
            text_surf = font.render(text, True, color)
            shadow_surf = font.render(text, True, self.COLOR_TEXT_SHADOW)
            text_rect = text_surf.get_rect()
            setattr(text_rect, align, (x, y))
            self.screen.blit(shadow_surf, text_rect.move(2, 2))
            self.screen.blit(text_surf, text_rect)

        # Score
        draw_text(f"SCORE: {self.score}", self.font_large, self.COLOR_TEXT, 20, 10)
        
        # Lives
        lives_text = "LIVES: " + "♥ " * self.lives
        draw_text(lives_text, self.font_large, self.COLOR_TEXT, self.WIDTH - 20, 10, align="topright")
        
        # Final platform indicator
        win_platform = self.platforms[-1]
        dist_to_win = self.player_pos[1] - win_platform.centery
        if dist_to_win > 0 and self.highest_platform_idx < self.NUM_PLATFORMS - 1:
             draw_text(f"GOAL: {int(dist_to_win)}m", self.font_small, self.COLOR_TEXT, self.WIDTH // 2, 10, align="midtop")

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "highest_platform": self.highest_platform_idx,
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

# Example of how to run the environment for manual play
if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    # env.validate_implementation() # this would fail without a reset first
    
    # --- Manual Play Example ---
    # This part requires a display. If running headlessly, comment it out.
    try:
        # Set a display driver. Use "dummy" for headless, or a specific one for display.
        # import os # Already imported at top
        if "SDL_VIDEODRIVER" not in os.environ or os.environ["SDL_VIDEODRIVER"] == "dummy":
             os.environ["SDL_VIDEODRIVER"] = "x11"
        
        screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        pygame.display.set_caption("Hopper Environment")
        
        obs, info = env.reset()
        done = False
        
        print("\n" + "="*30)
        print("MANUAL PLAY")
        print(env.game_description)
        print(env.user_guide)
        print("="*30 + "\n")

        while not done:
            # Action mapping for keyboard
            keys = pygame.key.get_pressed()
            movement = 0 # None
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [movement, space_held, shift_held]
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Render the observation to the display
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            # Handle Pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
            
            env.clock.tick(env.FPS)
            
        print("Game Over!")
        print(f"Final Info: {info}")
            
    except pygame.error as e:
        print(f"\nPygame display error: {e}")
        print("Skipping manual play example. The environment is likely running in a headless environment.")
        print("The core GameEnv class is still valid and can be used by RL agents.")
        
    finally:
        env.close()