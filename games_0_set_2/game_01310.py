
# Generated: 2025-08-27T16:43:36.480208
# Source Brief: brief_01310.md
# Brief Index: 1310

        
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
    """
    A minimalist side-scrolling arcade game where the player must precisely time jumps
    between procedurally generated platforms to reach the end within a time limit.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # User-facing strings
    user_guide = (
        "Controls: Press space to jump. The player moves forward automatically."
    )
    game_description = (
        "Minimalist side-scrolling platformer. Time your jumps precisely to "
        "reach the final platform before the timer runs out. Good luck!"
    )

    # Frame advance setting
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    TIME_LIMIT_SECONDS = 30

    # Physics
    PLAYER_SPEED = 6
    GRAVITY = 1.0
    JUMP_STRENGTH = -15  # Negative is up
    PLAYER_SIZE = 20

    # Game rules
    NUM_PLATFORMS = 10
    MAX_STEPS = FPS * (TIME_LIMIT_SECONDS + 5) # Time limit + 5s buffer

    # Colors
    COLOR_BG_TOP = (10, 20, 40)
    COLOR_BG_BOTTOM = (0, 0, 0)
    COLOR_PLAYER = (255, 64, 64)
    COLOR_PLATFORM = (240, 240, 240)
    COLOR_GOAL = (64, 255, 64)
    COLOR_UI_TEXT = (255, 255, 255)
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        self.render_mode = render_mode

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_big = pygame.font.SysFont("monospace", 48, bold=True)

        # Initialize state variables to avoid attribute errors before reset
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = [0, 0]
        self.player_vel_y = 0.0
        self.on_ground = False
        self.platforms = []
        self.current_platform_index = 0
        self.successful_jumps = 0
        self.time_left = 0.0
        self.particles = []
        self.last_space_held = False
        self.camera_x = 0.0
        self.termination_reason = ""

        # Validate implementation after setup
        # self.validate_implementation() # Optional: call for debugging

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_left = self.TIME_LIMIT_SECONDS
        self.termination_reason = ""
        
        # Reset player
        self.player_pos = [100.0, 200.0]
        self.player_vel_y = 0.0
        self.on_ground = True
        
        # Reset progression
        self.current_platform_index = 0
        self.successful_jumps = 0

        # Reset input state
        self.last_space_held = False

        # Reset world elements
        self._generate_platforms()
        self.particles = []
        self.camera_x = 0.0

        return self._get_observation(), self._get_info()

    def _generate_platforms(self):
        self.platforms = []
        # Initial platform
        start_platform = pygame.Rect(50, 250, 150, 20)
        self.platforms.append(start_platform)
        
        last_platform = start_platform
        
        for i in range(self.NUM_PLATFORMS - 1):
            gap_difficulty_modifier = (self.successful_jumps // 5) * 0.5
            gap_x = self.np_random.uniform(50 + gap_difficulty_modifier, 100 + gap_difficulty_modifier)
            
            max_y_up = 100
            max_y_down = 50
            gap_y = self.np_random.uniform(-max_y_up, max_y_down)

            new_x = last_platform.right + gap_x
            new_y = np.clip(last_platform.y + gap_y, 100, self.SCREEN_HEIGHT - 50)
            new_width = self.np_random.uniform(80, 150)
            
            new_platform = pygame.Rect(new_x, new_y, new_width, 20)
            self.platforms.append(new_platform)
            last_platform = new_platform

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack action
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0.0

        # --- Game Logic ---
        self._handle_input(space_held)
        self._update_player_physics()
        reward += self._handle_collisions_and_progress()
        self._update_particles()

        # Update timers and step count
        self.time_left -= 1.0 / self.FPS
        self.steps += 1
        
        # Check for termination
        terminated, term_reward = self._check_termination()
        reward += term_reward
        self.game_over = terminated

        # Final return tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated is always False
            self._get_info()
        )
    
    def _handle_input(self, space_held):
        # Jump on space press (rising edge)
        if space_held and not self.last_space_held and self.on_ground:
            self.player_vel_y = self.JUMP_STRENGTH
            self.on_ground = False
            # Sound: Jump sfx
        self.last_space_held = space_held

    def _update_player_physics(self):
        # Horizontal movement is constant
        self.player_pos[0] += self.PLAYER_SPEED
        
        # Vertical movement (gravity)
        if not self.on_ground:
            self.player_vel_y += self.GRAVITY
        self.player_pos[1] += self.player_vel_y

    def _handle_collisions_and_progress(self):
        reward = 0.0
        player_rect = pygame.Rect(self.player_pos[0], self.player_pos[1], self.PLAYER_SIZE, self.PLAYER_SIZE)
        
        # Assume not on ground until a collision proves otherwise
        landed_this_frame = False
        
        for i, platform in enumerate(self.platforms):
            # Check for collision and if player is falling onto it from above
            if player_rect.colliderect(platform) and self.player_vel_y > 0:
                # Check if the player's bottom was above the platform top in the previous frame
                if (player_rect.bottom - self.player_vel_y) <= platform.top + 1:
                    self.player_pos[1] = platform.top - self.PLAYER_SIZE
                    self.player_vel_y = 0
                    self.on_ground = True
                    landed_this_frame = True
                    
                    # Check for landing on a new platform
                    if i > self.current_platform_index:
                        reward += 1.0  # Reward for new platform
                        self.score += 1
                        self.successful_jumps += 1
                        self.current_platform_index = i
                        self._create_landing_particles(player_rect.midbottom)
                        # Sound: Land sfx
                        
                        # Check for difficulty increase
                        if self.successful_jumps > 0 and self.successful_jumps % 5 == 0:
                            # In a real game, you might regenerate future platforms here
                            pass

                    break # Stop checking after first valid landing
        
        if self.on_ground and landed_this_frame:
            reward += 0.1 # Reward for being on a platform
            
        return reward

    def _check_termination(self):
        terminated = False
        terminal_reward = 0.0

        # Win condition
        if self.current_platform_index >= self.NUM_PLATFORMS - 1:
            terminated = True
            terminal_reward = 100.0
            self.termination_reason = "YOU WIN!"

        # Lose conditions
        elif self.player_pos[1] > self.SCREEN_HEIGHT + 50: # Fell off
            terminated = True
            terminal_reward = -100.0
            self.termination_reason = "GAME OVER"
        elif self.time_left <= 0: # Timeout
            terminated = True
            terminal_reward = -100.0
            self.termination_reason = "TIME UP"
        elif self.steps >= self.MAX_STEPS: # Step limit
            terminated = True
            terminal_reward = -100.0
            self.termination_reason = "STEP LIMIT"

        return terminated, terminal_reward

    def _get_observation(self):
        # Smooth camera follows player
        self.camera_x += (self.player_pos[0] - self.camera_x - self.SCREEN_WIDTH / 3) * 0.1

        self._render_background()
        self._render_game_elements()
        self._render_particles()
        self._render_player()
        self._render_ui()
        
        # Convert to numpy array (required format)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        # Draw a vertical gradient
        for y in range(self.SCREEN_HEIGHT):
            ratio = y / self.SCREEN_HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - ratio) + self.COLOR_BG_BOTTOM[0] * ratio,
                self.COLOR_BG_TOP[1] * (1 - ratio) + self.COLOR_BG_BOTTOM[1] * ratio,
                self.COLOR_BG_TOP[2] * (1 - ratio) + self.COLOR_BG_BOTTOM[2] * ratio,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))

    def _render_game_elements(self):
        # Render platforms relative to camera
        for i, platform in enumerate(self.platforms):
            color = self.COLOR_GOAL if i == self.NUM_PLATFORMS - 1 else self.COLOR_PLATFORM
            render_rect = platform.copy()
            render_rect.x -= int(self.camera_x)
            pygame.draw.rect(self.screen, color, render_rect, border_radius=3)

    def _render_player(self):
        player_render_x = int(self.player_pos[0] - self.camera_x)
        player_render_y = int(self.player_pos[1])
        player_rect = pygame.Rect(player_render_x, player_render_y, self.PLAYER_SIZE, self.PLAYER_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=4)
        # Add a small "eye" for direction
        eye_x = player_rect.centerx + 5
        eye_y = player_rect.centery - 4
        pygame.draw.circle(self.screen, (255,255,255), (eye_x, eye_y), 3)

    def _render_particles(self):
        for p in self.particles:
            pos_x = int(p['pos'][0] - self.camera_x)
            pos_y = int(p['pos'][1])
            radius = int(p['size'])
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, pos_x, pos_y, radius, p['color'])

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['size'] -= 0.2
        self.particles = [p for p in self.particles if p['size'] > 0]

    def _create_landing_particles(self, pos):
        for _ in range(15):
            angle = self.np_random.uniform(math.pi, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'size': self.np_random.uniform(2, 5),
                'color': (200, 200, 200, 150)
            })

    def _render_ui(self):
        # Timer
        timer_text = f"TIME: {max(0, self.time_left):.1f}"
        timer_surface = self.font_ui.render(timer_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(timer_surface, (10, 10))

        # Platform count
        platform_text = f"PLATFORM: {self.current_platform_index + 1}/{self.NUM_PLATFORMS}"
        platform_surface = self.font_ui.render(platform_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(platform_surface, (self.SCREEN_WIDTH - platform_surface.get_width() - 10, 10))

        # Game over text
        if self.game_over:
            end_text_surface = self.font_big.render(self.termination_reason, True, self.COLOR_UI_TEXT)
            text_rect = end_text_surface.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text_surface, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": self.time_left,
            "current_platform": self.current_platform_index,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        print("🔬 Validating implementation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space
        self.reset()
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), f"Obs shape is {test_obs.shape}"
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example usage for human play
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # --- Pygame setup for human play ---
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption(GameEnv.game_description)
    clock = pygame.time.Clock()

    obs, info = env.reset()
    done = False
    
    print("\n" + "="*30)
    print("      HUMAN PLAY MODE")
    print("="*30)
    print(env.user_guide)
    print("Press R to reset the game.")
    
    while not done:
        # --- Action mapping for human play ---
        keys = pygame.key.get_pressed()
        space_held = keys[pygame.K_SPACE]
        
        # The MultiDiscrete action is [movement, space, shift]
        # We only care about space for this game.
        action = [0, 1 if space_held else 0, 0]

        # --- Environment step ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Rendering ---
        # Convert the observation (H, W, C) to a Pygame surface (W, H)
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # --- Event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting game...")
                obs, info = env.reset()

        if terminated or truncated:
            # Wait for a moment on the game over screen before resetting
            pygame.time.wait(2000)
            obs, info = env.reset()

        clock.tick(GameEnv.FPS)
        
    env.close()
    print("Game window closed.")