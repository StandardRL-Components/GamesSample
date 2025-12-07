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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move the falling block. Press space to drop it quickly. Stack 20 to win!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Stack falling blocks as high as possible. A stable stack is key. Win by stacking 20 blocks, or lose if the stack collapses or time runs out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.TIME_LIMIT_SECONDS = 60
        self.MAX_STEPS = self.TIME_LIMIT_SECONDS * self.FPS

        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (40, 45, 60)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_SCORE = (255, 215, 0)
        self.COLOR_TIMER = (255, 80, 80)
        self.BLOCK_COLORS = [
            (255, 70, 70),  # Red
            (70, 255, 70),  # Green
            (70, 130, 255), # Blue
            (255, 255, 70), # Yellow
            (255, 70, 255), # Magenta
            (70, 255, 255), # Cyan
        ]

        # Game Area
        self.PLAY_AREA_WIDTH = 240
        self.PLAY_AREA_X = (self.WIDTH - self.PLAY_AREA_WIDTH) // 2
        self.BASE_HEIGHT = 20

        # Block Properties
        self.BLOCK_WIDTH = 60
        self.BLOCK_HEIGHT = 20
        self.BLOCK_MOVE_SPEED = 6
        self.DROP_SPEED = 15
        self.INITIAL_GRAVITY = 0.2

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
        self.font_ui = pygame.font.Font(None, 28)
        self.font_height = pygame.font.Font(None, 24)
        self.font_game_over = pygame.font.Font(None, 60)

        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.np_random = None
        
        self.stacked_blocks = []
        self.falling_block = None
        self.particles = []
        self.current_gravity = self.INITIAL_GRAVITY
        self.space_was_held = False
        self.last_stack_size_for_reward = 0
        self.win_condition_met = False
        self.collapse_reason = ""

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_condition_met = False
        self.collapse_reason = ""

        self.stacked_blocks = []
        self.particles = []
        self.current_gravity = self.INITIAL_GRAVITY
        self.space_was_held = False
        self.last_stack_size_for_reward = 0
        
        # Create the base platform
        base_rect = pygame.Rect(
            self.PLAY_AREA_X,
            self.HEIGHT - self.BASE_HEIGHT,
            self.PLAY_AREA_WIDTH,
            self.BASE_HEIGHT,
        )
        self.stacked_blocks.append({
            "rect": base_rect,
            "color": (100, 100, 120),
            "stable": True,
        })
        
        self._spawn_new_block()

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        terminated = False

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.steps += 1

        # --- Handle Action ---
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        if self.falling_block:
            # Horizontal movement
            if movement == 3:  # Left
                self.falling_block["pos"][0] -= self.BLOCK_MOVE_SPEED
                reward -= 0.02
            elif movement == 4:  # Right
                self.falling_block["pos"][0] += self.BLOCK_MOVE_SPEED
                reward -= 0.02
            
            # Clamp position to play area
            self.falling_block["pos"][0] = max(
                self.PLAY_AREA_X,
                min(self.falling_block["pos"][0], self.PLAY_AREA_X + self.PLAY_AREA_WIDTH - self.BLOCK_WIDTH)
            )

            # Drop action (on press, not hold)
            if space_held and not self.space_was_held:
                self.falling_block["vel"][1] = self.DROP_SPEED
        
        self.space_was_held = space_held

        # --- Game Logic & Physics ---
        self.current_gravity = self.INITIAL_GRAVITY + 0.01 * (self.steps // 100)

        if self.falling_block:
            # Apply gravity
            self.falling_block["vel"][1] += self.current_gravity
            self.falling_block["pos"][1] += self.falling_block["vel"][1]

            # Update rect for collision detection
            falling_rect = pygame.Rect(
                int(self.falling_block["pos"][0]),
                int(self.falling_block["pos"][1]),
                self.BLOCK_WIDTH,
                self.BLOCK_HEIGHT
            )

            # Collision check
            for stacked_block in self.stacked_blocks:
                if falling_rect.colliderect(stacked_block["rect"]):
                    # Collision happened, place the block
                    falling_rect.bottom = stacked_block["rect"].top
                    
                    newly_stacked_block = {
                        "rect": falling_rect,
                        "color": self.falling_block["color"],
                        "stable": True,
                    }

                    is_stable = self._check_stability(newly_stacked_block)
                    self.stacked_blocks.append(newly_stacked_block)

                    if is_stable:
                        reward += 0.1 # Successful placement
                        self._create_particles(falling_rect.midbottom, self.falling_block["color"], 20, 2)
                        
                        stack_count = len(self.stacked_blocks) - 1 # Exclude base
                        if stack_count > 0 and stack_count % 3 == 0 and stack_count > self.last_stack_size_for_reward:
                            reward += 1.0
                            self.last_stack_size_for_reward = stack_count

                        if stack_count >= 20:
                            reward += 100
                            self.game_over = True
                            self.win_condition_met = True
                            self.collapse_reason = "YOU WIN!"
                        else:
                            self._spawn_new_block()
                    else:
                        # Unstable, game over
                        reward -= 50
                        self.game_over = True
                        self.collapse_reason = "STACK COLLAPSED!"
                        newly_stacked_block["stable"] = False
                        self._create_particles(falling_rect.center, (200, 200, 200), 50, 4)
                    
                    self.falling_block = None
                    break # Exit collision check loop

        # --- Termination Conditions ---
        if not self.game_over and self.steps >= self.MAX_STEPS:
            reward -= 10
            self.game_over = True
            self.collapse_reason = "TIME'S UP!"

        self.score += reward
        terminated = self.game_over
        truncated = False # Time limit is a termination condition in this game logic

        return self._get_observation(), reward, terminated, truncated, self._get_info()
    
    def _spawn_new_block(self):
        start_x = self.np_random.integers(
            self.PLAY_AREA_X, self.PLAY_AREA_X + self.PLAY_AREA_WIDTH - self.BLOCK_WIDTH + 1
        )
        color_index = self.np_random.integers(0, len(self.BLOCK_COLORS))
        self.falling_block = {
            "pos": [float(start_x), -float(self.BLOCK_HEIGHT)],
            "vel": [0.0, 0.0],
            "color": self.BLOCK_COLORS[color_index],
        }

    def _check_stability(self, new_block_data):
        new_rect = new_block_data["rect"]
        support_min_x = float('inf')
        support_max_x = float('-inf')
        found_support = False

        for block in self.stacked_blocks:
            if block is new_block_data: continue # Don't check against itself
            stacked_rect = block["rect"]
            # Check for horizontal overlap and vertical contact
            is_horiz_overlap = (new_rect.left < stacked_rect.right and new_rect.right > stacked_rect.left)
            is_vertical_contact = abs(new_rect.bottom - stacked_rect.top) < 5 # 5px tolerance

            if is_horiz_overlap and is_vertical_contact:
                found_support = True
                support_min_x = min(support_min_x, stacked_rect.left)
                support_max_x = max(support_max_x, stacked_rect.right)

        if not found_support:
            return False # Fell off the side

        # Check if center of mass is within support bounds
        return new_rect.centerx >= support_min_x and new_rect.centerx <= support_max_x

    def _update_and_draw_particles(self):
        active_particles = []
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["life"] -= 1

            px, py = int(p["pos"][0]), int(p["pos"][1])

            # Only process/draw particles that are alive and on-screen
            if p["life"] > 0 and 0 <= px < self.WIDTH and 0 <= py < self.HEIGHT:
                active_particles.append(p)
                alpha = max(0, min(255, int(255 * (p["life"] / p["max_life"]))))
                
                # Pygame.gfxdraw doesn't support alpha in filled_circle, so we fake it by blending
                bg_slice = self.screen.get_at((px, py))
                blend_ratio = alpha / 255.0
                draw_color = (
                    int(p["color"][0] * blend_ratio + bg_slice[0] * (1 - blend_ratio)),
                    int(p["color"][1] * blend_ratio + bg_slice[1] * (1 - blend_ratio)),
                    int(p["color"][2] * blend_ratio + bg_slice[2] * (1 - blend_ratio)),
                )
                
                size = int(p["size"] * (p["life"] / p["max_life"]))
                if size > 0:
                    pygame.gfxdraw.filled_circle(self.screen, px, py, size, draw_color)
        
        self.particles = active_particles

    def _create_particles(self, pos, color, count, speed_factor):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3) * speed_factor
            self.particles.append({
                "pos": list(pos),
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "color": color,
                "life": self.np_random.integers(15, 30),
                "max_life": 30,
                "size": self.np_random.integers(2, 5),
            })

    def _get_observation(self):
        # --- Background ---
        self.screen.fill(self.COLOR_BG)
        for x in range(0, self.WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))
        
        # --- Game Elements ---
        # Draw stacked blocks
        for i, block_data in enumerate(self.stacked_blocks):
            rect = block_data["rect"]
            base_color = block_data["color"]
            
            # Fade older blocks (lower in the stack)
            fade_factor = max(0.4, 1.0 - (len(self.stacked_blocks) - i) * 0.03)
            color = tuple(int(c * fade_factor) for c in base_color)
            
            if not block_data["stable"]: # Unstable blocks are red
                color = (200, 50, 50)
            
            pygame.draw.rect(self.screen, color, rect)
            pygame.draw.rect(self.screen, tuple(min(255, c+30) for c in color), rect, 2)

        # Draw falling block
        if self.falling_block:
            rect = pygame.Rect(
                int(self.falling_block["pos"][0]),
                int(self.falling_block["pos"][1]),
                self.BLOCK_WIDTH,
                self.BLOCK_HEIGHT
            )
            color = self.falling_block["color"]
            pygame.draw.rect(self.screen, color, rect)
            pygame.draw.rect(self.screen, tuple(min(255, c+50) for c in color), rect, 3)

        # Draw particles
        self._update_and_draw_particles()

        # --- UI Overlay ---
        # Score
        score_text = self.font_ui.render(f"SCORE: {int(self.score)}", True, self.COLOR_SCORE)
        self.screen.blit(score_text, (10, 10))

        # Timer
        time_left = (self.MAX_STEPS - self.steps) / self.FPS
        timer_text = self.font_ui.render(f"TIME: {max(0, time_left):.1f}", True, self.COLOR_TIMER)
        self.screen.blit(timer_text, (self.WIDTH - timer_text.get_width() - 10, 10))

        # Height
        stack_height = len(self.stacked_blocks) - 1
        height_text = self.font_height.render(f"HEIGHT: {stack_height} / 20", True, self.COLOR_TEXT)
        height_pos = (self.WIDTH // 2 - height_text.get_width() // 2, self.HEIGHT - 30)
        self.screen.blit(height_text, height_pos)

        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            end_text = self.font_game_over.render(self.collapse_reason, True, self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(end_text, text_rect)

        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stack_height": len(self.stacked_blocks) - 1,
            "is_game_over": self.game_over,
        }

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    env = GameEnv()
    
    # --- Manual Play Loop ---
    obs, info = env.reset()
    done = False
    
    # Set up a window to display the game
    pygame.display.set_caption("Block Stacker")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0  # 0=none
        space = 0     # 0=released
        shift = 0     # 0=released
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
            
        if keys[pygame.K_SPACE]:
            space = 1
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift = 1

        if done:
            # Wait for a key press to reset
            if any(keys):
                obs, info = env.reset()
                done = False
                total_reward = 0
                pygame.time.wait(200) # Debounce
        else:
            action = np.array([movement, space, shift])
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

        # --- Rendering ---
        # The observation is already a rendered frame
        # We just need to convert it back to a Pygame surface to display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.FPS)
        
    env.close()