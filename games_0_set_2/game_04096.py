
# Generated: 2025-08-28T01:24:45.692771
# Source Brief: brief_04096.md
# Brief Index: 4096

        
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
        "Controls: ←→ to move the block. Press space to drop it."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Stack falling blocks as high as possible. Reach a height of 20 to win. The game ends if your tower topples or time runs out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 100 # As per brief's max_steps calculation
        self.MAX_TIME_SECONDS = 60
        self.MAX_STEPS = self.MAX_TIME_SECONDS * self.FPS

        # Colors
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.COLOR_BOUNDARY = (100, 100, 120)
        self.COLOR_TARGET_LINE = (0, 255, 128)
        self.BLOCK_COLORS = [
            (255, 99, 71),   # Tomato
            (70, 130, 180),  # SteelBlue
            (50, 205, 50),   # LimeGreen
            (255, 215, 0),   # Gold
            (147, 112, 219), # MediumPurple
        ]

        # Game Mechanics
        self.PLAY_AREA_WIDTH = 200
        self.BLOCK_UNIT_WIDTH = 20
        self.BLOCK_HEIGHT = 15
        self.WIN_HEIGHT = 20
        self.INITIAL_FALL_SPEED = 0.5 # pixels per step
        self.FALL_SPEED_INCREASE = 0.05
        self.PLAYER_MOVE_SPEED = 3 # pixels per step

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_ui = pygame.font.SysFont("monospace", 16)
            self.font_big = pygame.font.SysFont("monospace", 48, bold=True)
        except pygame.error:
            self.font_ui = pygame.font.Font(None, 20)
            self.font_big = pygame.font.Font(None, 60)
        
        # --- Game State Initialization ---
        self.rng = None
        self.play_area_rect = None
        self.floor_y = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.time_remaining = 0
        self.fall_speed = 0
        self.placements_since_speed_increase = 0
        self.current_height = 0
        self.stacked_blocks = [] # List of (pygame.Rect, color) tuples
        self.falling_block = {}  # Dict with 'rect', 'color', 'y_float'
        self.last_space_state = False
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        elif self.rng is None:
            self.rng = np.random.default_rng()
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.time_remaining = self.MAX_STEPS
        
        self.play_area_rect = pygame.Rect(
            (self.WIDTH - self.PLAY_AREA_WIDTH) // 2,
            50,
            self.PLAY_AREA_WIDTH,
            self.HEIGHT - 60
        )
        self.floor_y = self.play_area_rect.bottom

        self.stacked_blocks = []
        self.current_height = 0
        self.fall_speed = self.INITIAL_FALL_SPEED
        self.placements_since_speed_increase = 0
        self.last_space_state = False
        
        self._spawn_new_block()

        return self._get_observation(), self._get_info()

    def _spawn_new_block(self):
        width_units = self.rng.integers(1, 6)
        width_pixels = width_units * self.BLOCK_UNIT_WIDTH
        color_index = self.rng.integers(0, len(self.BLOCK_COLORS))
        color = self.BLOCK_COLORS[color_index]
        
        rect = pygame.Rect(
            self.play_area_rect.centerx - width_pixels // 2,
            self.play_area_rect.top,
            width_pixels,
            self.BLOCK_HEIGHT
        )
        
        self.falling_block = {
            'rect': rect,
            'color': color,
            'y_float': float(rect.y)
        }

    def step(self, action):
        reward = 0.0

        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        is_drop_action = space_held and not self.last_space_state
        self.last_space_state = space_held
        
        # Update game logic
        self.steps += 1
        self.time_remaining -= 1

        # --- Handle Player Input ---
        if movement == 3: # Left
            self.falling_block['rect'].x -= self.PLAYER_MOVE_SPEED
        elif movement == 4: # Right
            self.falling_block['rect'].x += self.PLAYER_MOVE_SPEED

        # Clamp block within screen bounds for movement
        self.falling_block['rect'].left = max(0, self.falling_block['rect'].left)
        self.falling_block['rect'].right = min(self.WIDTH, self.falling_block['rect'].right)

        collision_y = self._get_collision_y()

        # --- Drop Action or Natural Fall ---
        if is_drop_action:
            # Place block immediately
            self.falling_block['rect'].bottom = collision_y
            reward += self._place_block()
        else:
            # Natural fall
            self.falling_block['y_float'] += self.fall_speed
            self.falling_block['rect'].y = int(self.falling_block['y_float'])
            
            if self.falling_block['rect'].bottom >= collision_y:
                self.falling_block['rect'].bottom = collision_y
                reward += self._place_block()
        
        self.score += reward

        # --- Check Termination Conditions ---
        if self.time_remaining <= 0 and not self.game_over:
            self.game_over = True
            reward = -100.0 # Penalty for running out of time
            self.score += reward
        
        if self.current_height >= self.WIN_HEIGHT and not self.game_over:
            self.game_over = True
            self.win = True
            reward = 100.0 # Big reward for winning
            self.score += reward

        terminated = self.game_over or self.steps >= self.MAX_STEPS
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _get_collision_y(self):
        """Find the y-coordinate where the falling block would collide."""
        highest_support_y = self.floor_y
        falling_rect = self.falling_block['rect']
        
        for block_rect, _ in self.stacked_blocks:
            # Check for horizontal overlap
            if (falling_rect.left < block_rect.right and falling_rect.right > block_rect.left):
                highest_support_y = min(highest_support_y, block_rect.top)
        
        return highest_support_y

    def _place_block(self):
        """Handle logic for placing a block and calculating rewards."""
        reward = 0.0
        placed_rect = self.falling_block['rect']

        # 1. Find support surface
        support_blocks = [b for b, _ in self.stacked_blocks if b.top == placed_rect.bottom]
        
        min_support_x = self.play_area_rect.left
        max_support_x = self.play_area_rect.right
        
        if support_blocks:
            min_support_x = min(b.left for b in support_blocks)
            max_support_x = max(b.right for b in support_blocks)

        # 2. Check for loss condition (instability or out of bounds)
        if (placed_rect.left < min_support_x or 
            placed_rect.right > max_support_x or
            placed_rect.left < self.play_area_rect.left or
            placed_rect.right > self.play_area_rect.right):
            
            self.game_over = True
            self.stacked_blocks.append((placed_rect, (255, 0, 0))) # Render failed block
            # sound: block_fall_off.wav
            return -100.0 # Large penalty for losing
        
        # 3. Successful placement
        # sound: block_place.wav
        self.stacked_blocks.append((placed_rect.copy(), self.falling_block['color']))
        if self.stacked_blocks:
             self.current_height = (self.floor_y - min(b.top for b, _ in self.stacked_blocks)) // self.BLOCK_HEIGHT
        self.placements_since_speed_increase += 1
        
        reward += 0.1 # Small reward for successful placement

        # 4. Check for perfect placement bonus
        support_center_x = (min_support_x + max_support_x) / 2
        if abs(placed_rect.centerx - support_center_x) < 2:
            reward += 1.0 # Bonus for perfect centering
            # sound: perfect_place.wav

        # 5. Increase difficulty
        if self.placements_since_speed_increase >= 5:
            self.fall_speed += self.FALL_SPEED_INCREASE
            self.placements_since_speed_increase = 0

        # 6. Spawn next block
        self._spawn_new_block()
        
        return reward

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
        # Draw play area boundaries
        pygame.draw.line(self.screen, self.COLOR_BOUNDARY, self.play_area_rect.bottomleft, self.play_area_rect.bottomright, 2)
        pygame.draw.line(self.screen, self.COLOR_BOUNDARY, self.play_area_rect.topleft, self.play_area_rect.bottomleft, 1)
        pygame.draw.line(self.screen, self.COLOR_BOUNDARY, self.play_area_rect.topright, self.play_area_rect.bottomright, 1)

        # Draw target height line
        target_y = self.floor_y - (self.WIN_HEIGHT * self.BLOCK_HEIGHT)
        if target_y > self.play_area_rect.top:
            pygame.draw.line(self.screen, self.COLOR_TARGET_LINE, (self.play_area_rect.left, target_y), (self.play_area_rect.right, target_y), 2)

        # Draw stacked blocks
        for rect, color in self.stacked_blocks:
            pygame.draw.rect(self.screen, color, rect)
            pygame.draw.rect(self.screen, self.COLOR_BG, rect, 1) # Outline

        # Draw falling block with glow
        if not self.game_over:
            block = self.falling_block
            # Glow effect
            glow_rect = block['rect'].inflate(10, 10)
            glow_surface = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            glow_color = block['color'] + (60,) # Add alpha
            pygame.draw.rect(glow_surface, glow_color, glow_surface.get_rect(), border_radius=5)
            self.screen.blit(glow_surface, glow_rect.topleft)

            # Main block
            pygame.draw.rect(self.screen, block['color'], block['rect'])
            pygame.draw.rect(self.screen, tuple(min(c+50, 255) for c in self.COLOR_BG), block['rect'], 1)

            # Draw projection line to collision point
            proj_y = self._get_collision_y()
            start_pos = (block['rect'].centerx, block['rect'].bottom)
            end_pos = (block['rect'].centerx, proj_y)
            if end_pos[1] > start_pos[1]:
                # Use gfxdraw for anti-aliased line with alpha
                r, g, b = block['color']
                pygame.gfxdraw.line(self.screen, int(start_pos[0]), int(start_pos[1]), int(end_pos[0]), int(end_pos[1]), (r, g, b, 100))

    def _render_ui(self):
        # Height
        height_text = self.font_ui.render(f"Height: {self.current_height}/{self.WIN_HEIGHT}", True, self.COLOR_UI_TEXT)
        self.screen.blit(height_text, (10, 10))

        # Time
        time_str = f"Time: {max(0, self.time_remaining / self.FPS):.1f}"
        time_text = self.font_ui.render(time_str, True, self.COLOR_UI_TEXT)
        self.screen.blit(time_text, (self.WIDTH - time_text.get_width() - 10, 10))

        # Score
        score_text = self.font_ui.render(f"Score: {self.score:.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.WIDTH // 2 - score_text.get_width() // 2, self.HEIGHT - 30))
        
        # Game Over / Win message
        if self.game_over:
            msg = "YOU WIN!" if self.win else "GAME OVER"
            color = self.COLOR_TARGET_LINE if self.win else (255, 80, 80)
            
            msg_text = self.font_big.render(msg, True, color)
            msg_rect = msg_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2 - 20))
            
            # Simple shadow
            shadow_text = self.font_big.render(msg, True, (0,0,0))
            self.screen.blit(shadow_text, msg_rect.move(3,3))
            self.screen.blit(msg_text, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "height": self.current_height,
            "time_remaining_seconds": max(0, self.time_remaining / self.FPS),
        }

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # --- Manual Control Setup ---
    # action = [movement, space, shift]
    # movement: 0=none, 1=up, 2=down, 3=left, 4=right
    action = np.array([0, 0, 0]) 
    
    # --- Pygame window for human play ---
    render_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Block Stacker")
    clock = pygame.time.Clock()
    
    print("\n" + "="*30)
    print("      MANUAL PLAY MODE")
    print("="*30)
    print(env.user_guide)
    print("Press R to reset.")
    print("Press Q to quit.")
    print("="*30 + "\n")

    game_is_running = True
    while game_is_running:
        # --- Event Handling for Manual Control ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_is_running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    action.fill(0) # Reset action array
                    done = False
                    print("--- Game Reset ---")
                if event.key == pygame.K_q:
                    game_is_running = False

        if not done:
            # --- Key presses to action array ---
            keys = pygame.key.get_pressed()
            
            # Movement
            if keys[pygame.K_LEFT]:
                action[0] = 3
            elif keys[pygame.K_RIGHT]:
                action[0] = 4
            else:
                action[0] = 0 # No movement
            
            # Space bar
            action[1] = 1 if keys[pygame.K_SPACE] else 0
            
            # Shift key (unused in this game)
            action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

            # --- Gym Step ---
            obs, reward, terminated, truncated, info = env.step(action)
            if reward != 0:
                print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']:.1f}, Height: {info['height']}")

            if terminated:
                done = True
                print(f"--- Game Over --- Final Score: {info['score']:.1f}, Final Height: {info['height']}")

        # --- Rendering ---
        # The observation is already a rendered frame, so we just need to display it
        # Need to transpose back from (H, W, C) to (W, H, C) for pygame
        frame_to_render = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame_to_render)
        render_screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        # Cap the frame rate for human playability
        clock.tick(env.FPS)

    env.close()
    pygame.quit()