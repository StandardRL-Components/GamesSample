import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A block-pushing puzzle game where the player must push colored blocks onto their
    matching targets against the clock. The game emphasizes planning and spatial reasoning.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrows to push selected block. Space to cycle selection clockwise, "
        "Shift to cycle counter-clockwise."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Push colored blocks onto their matching targets. Plan your moves carefully "
        "as you only have a few failed pushes and limited time to solve the puzzle."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    SCREEN_W, SCREEN_H = 640, 400
    FPS = 30
    
    # Grid and layout
    CELL_SIZE = 32
    GRID_W, GRID_H = 18, 11
    PLAY_AREA_W = GRID_W * CELL_SIZE
    PLAY_AREA_H = GRID_H * CELL_SIZE
    OFFSET_X = (SCREEN_W - PLAY_AREA_W) // 2
    OFFSET_Y = (SCREEN_H - PLAY_AREA_H) // 2
    
    # Game parameters
    NUM_BLOCKS = 15
    MAX_FAILS = 3
    TIME_LIMIT_SECONDS = 90
    MAX_STEPS = TIME_LIMIT_SECONDS * FPS
    
    # Colors (Clean, high-contrast palette)
    COLOR_BG = (20, 25, 40)
    COLOR_GRID = (40, 50, 70)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_OVERLAY = (20, 25, 40, 200)
    COLOR_WIN = (160, 255, 180)
    COLOR_LOSE = (255, 160, 160)
    
    BLOCK_COLORS = [
        (255, 100, 100), (100, 255, 100), (100, 100, 255),
        (255, 255, 100), (255, 100, 255), (100, 255, 255),
        (255, 180, 100), (180, 100, 255), (100, 255, 180),
        (255, 220, 180), (180, 255, 220), (220, 180, 255),
        (250, 150, 50), (50, 250, 150), (150, 50, 250)
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_H, self.SCREEN_W, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_W, self.SCREEN_H))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_big = pygame.font.SysFont("monospace", 50, bold=True)
        
        # Game state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_left = 0
        self.failed_attempts = 0
        self.blocks = []
        self.selected_block_idx = 0
        self.prev_space_held = False
        self.prev_shift_held = False
        self.particles = []
        self.screen_shake = 0
        
        # Initialize state variables
        self.reset()
        
        # Run self-check
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_left = self.MAX_STEPS
        self.failed_attempts = 0
        self.selected_block_idx = 0
        self.prev_space_held = False
        self.prev_shift_held = False
        self.particles = []
        self.screen_shake = 0
        
        self._generate_puzzle()
        
        return self._get_observation(), self._get_info()
    
    def _generate_puzzle(self):
        """Generates a new puzzle layout."""
        self.blocks = []
        all_coords = [(x, y) for x in range(self.GRID_W) for y in range(self.GRID_H)]
        self.np_random.shuffle(all_coords)

        # Ensure puzzle is not solved from the start
        while True:
            coords_copy = list(all_coords)
            block_coords = [coords_copy.pop(0) for _ in range(self.NUM_BLOCKS)]
            target_coords = [coords_copy.pop(0) for _ in range(self.NUM_BLOCKS)]
            
            is_solved = all(bc == tc for bc, tc in zip(block_coords, target_coords))
            if not is_solved:
                break
            # If by chance it's solved, reshuffle and try again
            self.np_random.shuffle(all_coords)

        for i in range(self.NUM_BLOCKS):
            pos = list(block_coords[i])
            target_pos = list(target_coords[i])
            visual_pos = [
                self.OFFSET_X + pos[0] * self.CELL_SIZE,
                self.OFFSET_Y + pos[1] * self.CELL_SIZE
            ]
            self.blocks.append({
                "id": i,
                "pos": pos,
                "target_pos": target_pos,
                "color": self.BLOCK_COLORS[i],
                "visual_pos": visual_pos,
                "on_target": pos == target_pos
            })

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.time_left -= 1
        reward = 0

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- Handle Input ---
        # Cycle selection on key press (transition from not held to held)
        if space_held and not self.prev_space_held:
            self.selected_block_idx = (self.selected_block_idx + 1) % self.NUM_BLOCKS
            # sfx: selection_change.wav
        if shift_held and not self.prev_shift_held:
            self.selected_block_idx = (self.selected_block_idx - 1 + self.NUM_BLOCKS) % self.NUM_BLOCKS
            # sfx: selection_change.wav
        
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        # --- Handle Block Pushing ---
        if movement != 0:
            reward += self._handle_push(movement)

        # --- Update block animations ---
        for block in self.blocks:
            target_vx = self.OFFSET_X + block["pos"][0] * self.CELL_SIZE
            target_vy = self.OFFSET_Y + block["pos"][1] * self.CELL_SIZE
            block["visual_pos"][0] += (target_vx - block["visual_pos"][0]) * 0.4
            block["visual_pos"][1] += (target_vy - block["visual_pos"][1]) * 0.4

        # --- Update particles ---
        self._update_particles()
        
        # --- Check for termination ---
        terminated, term_reward = self._check_termination()
        reward += term_reward
        self.game_over = terminated
        
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_push(self, movement):
        """Processes a push action, including chain reactions."""
        direction_map = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)} # Up, Down, Left, Right
        dx, dy = direction_map[movement]
        
        chain = []
        current_pos = self.blocks[self.selected_block_idx]["pos"]
        is_push_possible = True
        
        # 1. Determine the chain of blocks to be pushed
        temp_pos = list(current_pos)
        while True:
            block_idx = self._get_block_at(temp_pos)
            if block_idx is None: break # Should not happen on first iteration
            
            chain.append(block_idx)
            next_pos = (temp_pos[0] + dx, temp_pos[1] + dy)
            
            if not (0 <= next_pos[0] < self.GRID_W and 0 <= next_pos[1] < self.GRID_H):
                is_push_possible = False # Hit a wall
                break
            
            if self._get_block_at(next_pos) is None:
                break # Path is clear
            
            temp_pos[0] += dx
            temp_pos[1] += dy
        
        # 2. Execute the push or handle failure
        if not is_push_possible:
            self.failed_attempts += 1
            self.screen_shake = 10
            # sfx: push_fail.wav
            return -2.0 # Penalty for failed push
        else:
            push_reward = 0
            # sfx: push_success.wav
            for block_idx in reversed(chain):
                block = self.blocks[block_idx]
                
                # Reward for distance change
                old_dist = abs(block["pos"][0] - block["target_pos"][0]) + abs(block["pos"][1] - block["target_pos"][1])
                
                block["pos"][0] += dx
                block["pos"][1] += dy
                
                new_dist = abs(block["pos"][0] - block["target_pos"][0]) + abs(block["pos"][1] - block["target_pos"][1])
                
                if new_dist < old_dist: push_reward += 0.1
                elif new_dist > old_dist: push_reward -= 0.1
                
                # Check if block is now on target
                is_on_target = block["pos"] == block["target_pos"]
                if is_on_target and not block["on_target"]:
                    push_reward += 5.0 # Reward for placing a block
                    self._create_particles(block)
                    # sfx: target_achieved.wav
                block["on_target"] = is_on_target
            return push_reward

    def _check_termination(self):
        """Checks for win/loss conditions and returns reward."""
        if self.failed_attempts >= self.MAX_FAILS:
            return True, -10.0 # Loss: too many fails
        if self.time_left <= 0:
            return True, -10.0 # Loss: out of time
        
        if all(b["on_target"] for b in self.blocks):
            return True, 50.0 # Win: all blocks on targets
        
        return False, 0.0

    def _get_observation(self):
        # Determine screen shake offset
        shake_offset = (0, 0)
        if self.screen_shake > 0:
            self.screen_shake -= 1
            shake_offset = (self.np_random.integers(-4, 5), self.np_random.integers(-4, 5))

        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements to a temporary surface for shaking
        game_surface = pygame.Surface((self.SCREEN_W, self.SCREEN_H), pygame.SRCALPHA)
        self._render_game(game_surface)
        self.screen.blit(game_surface, shake_offset)
        
        # Render UI overlay (does not shake)
        self._render_ui(self.screen)
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self, surface):
        """Renders grid, targets, and blocks."""
        # Draw grid lines
        for x in range(self.GRID_W + 1):
            start = (self.OFFSET_X + x * self.CELL_SIZE, self.OFFSET_Y)
            end = (self.OFFSET_X + x * self.CELL_SIZE, self.OFFSET_Y + self.PLAY_AREA_H)
            pygame.draw.line(surface, self.COLOR_GRID, start, end, 1)
        for y in range(self.GRID_H + 1):
            start = (self.OFFSET_X, self.OFFSET_Y + y * self.CELL_SIZE)
            end = (self.OFFSET_X + self.PLAY_AREA_W, self.OFFSET_Y + y * self.CELL_SIZE)
            pygame.draw.line(surface, self.COLOR_GRID, start, end, 1)

        # Draw targets
        for block in self.blocks:
            tx, ty = block["target_pos"]
            color = block["color"]
            target_rect = pygame.Rect(
                self.OFFSET_X + tx * self.CELL_SIZE,
                self.OFFSET_Y + ty * self.CELL_SIZE,
                self.CELL_SIZE, self.CELL_SIZE
            )
            # Desaturate color for the target
            target_color = tuple(int(c * 0.4) for c in color)
            pygame.draw.rect(surface, target_color, target_rect)
            if block["on_target"]:
                 # Draw a checkmark-like shape
                pygame.draw.line(surface, self.COLOR_WIN, (target_rect.left+5, target_rect.centery), (target_rect.centerx-2, target_rect.bottom-5), 3)
                pygame.draw.line(surface, self.COLOR_WIN, (target_rect.centerx-2, target_rect.bottom-5), (target_rect.right-5, target_rect.top+5), 3)

        # Draw particles
        for p in self.particles:
            pygame.draw.circle(surface, p['color'], p['pos'], int(p['size']))

        # Draw blocks
        for i, block in enumerate(self.blocks):
            vx, vy = block["visual_pos"]
            block_rect = pygame.Rect(vx, vy, self.CELL_SIZE, self.CELL_SIZE)
            
            # Draw main block color
            pygame.draw.rect(surface, block["color"], block_rect, border_radius=4)
            # Draw a subtle border for definition
            border_color = tuple(min(255, c + 40) for c in block["color"])
            pygame.draw.rect(surface, border_color, block_rect, 2, border_radius=4)

        # Draw selection indicator
        selected_block = self.blocks[self.selected_block_idx]
        sel_vx, sel_vy = selected_block["visual_pos"]
        pulse = (math.sin(self.steps * 0.3) + 1) / 2 # 0 to 1
        indicator_size = self.CELL_SIZE + 6 + pulse * 4
        indicator_alpha = 150 + pulse * 105
        
        indicator_rect = pygame.Rect(
            sel_vx + self.CELL_SIZE / 2 - indicator_size / 2,
            sel_vy + self.CELL_SIZE / 2 - indicator_size / 2,
            indicator_size, indicator_size
        )
        
        # Draw a filled, rounded rectangle for the selection indicator.
        # This replaces the incorrect call to a non-existent pygame.gfxdraw.rounded_rectangle.
        # pygame.draw.rect supports the border_radius argument and alpha on SRALPHA surfaces.
        pygame.draw.rect(
            surface,
            (255, 255, 255, int(indicator_alpha)),
            indicator_rect,
            border_radius=8
        )

    def _render_ui(self, surface):
        """Renders time, fails, and game over messages."""
        # Time remaining
        time_str = f"TIME: {self.time_left / self.FPS:.1f}"
        time_surf = self.font_ui.render(time_str, True, self.COLOR_UI_TEXT)
        surface.blit(time_surf, (20, 10))

        # Failed attempts
        fail_str = f"FAILS: {self.failed_attempts}/{self.MAX_FAILS}"
        fail_surf = self.font_ui.render(fail_str, True, self.COLOR_UI_TEXT)
        surface.blit(fail_surf, (self.SCREEN_W - fail_surf.get_width() - 20, 10))
        
        # Game Over / Win message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_W, self.SCREEN_H), pygame.SRCALPHA)
            overlay.fill(self.COLOR_OVERLAY)
            
            is_win = all(b["on_target"] for b in self.blocks)
            text = "PUZZLE SOLVED!" if is_win else "GAME OVER"
            color = self.COLOR_WIN if is_win else self.COLOR_LOSE
            
            text_surf = self.font_big.render(text, True, color)
            text_rect = text_surf.get_rect(center=(self.SCREEN_W / 2, self.SCREEN_H / 2))
            overlay.blit(text_surf, text_rect)
            surface.blit(overlay, (0, 0))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": self.time_left,
            "failed_attempts": self.failed_attempts,
            "blocks_on_target": sum(1 for b in self.blocks if b["on_target"])
        }

    def _get_block_at(self, pos):
        """Returns the index of the block at a grid position, or None."""
        for i, block in enumerate(self.blocks):
            if block["pos"] == list(pos):
                return i
        return None

    def _create_particles(self, block):
        """Creates a burst of particles for a block reaching its target."""
        center_x = self.OFFSET_X + block["pos"][0] * self.CELL_SIZE + self.CELL_SIZE / 2
        center_y = self.OFFSET_Y + block["pos"][1] * self.CELL_SIZE + self.CELL_SIZE / 2
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.np_random.integers(15, 30)
            color = tuple(min(255, c+50) for c in block['color'])
            self.particles.append({
                "pos": [center_x, center_y],
                "vel": vel,
                "life": life,
                "max_life": life,
                "color": color,
                "size": self.np_random.uniform(2, 5)
            })

    def _update_particles(self):
        """Updates position, life, and size of all particles."""
        active_particles = []
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][0] *= 0.95 # friction
            p["vel"][1] *= 0.95
            p["life"] -= 1
            p["size"] = max(0, p["size"] * (p["life"] / p["max_life"]))
            if p["life"] > 0:
                active_particles.append(p)
        self.particles = active_particles

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
        assert test_obs.shape == (self.SCREEN_H, self.SCREEN_W, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_H, self.SCREEN_W, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_H, self.SCREEN_W, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example usage to run and visualize the game
if __name__ == '__main__':
    # The main loop is for visualization and interactive testing.
    # It requires a display, so we unset the dummy video driver if it was set.
    if os.environ.get("SDL_VIDEODRIVER") == "dummy":
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv()
    obs, info = env.reset()
    
    # --- Pygame setup for visualization ---
    screen = pygame.display.set_mode((GameEnv.SCREEN_W, GameEnv.SCREEN_H))
    pygame.display.set_caption("Block Pusher")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        # --- Action mapping from keyboard ---
        keys = pygame.key.get_pressed()
        movement = 0
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = keys[pygame.K_SPACE]
        shift_held = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]
        
        action = [movement, 1 if space_held else 0, 1 if shift_held else 0]
        
        # --- Environment step ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Rendering ---
        # The observation is already a rendered frame, so we just display it.
        # Pygame uses (width, height), but numpy uses (height, width), so transpose.
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f}")
            pygame.time.wait(3000) # Pause for 3 seconds
            env.reset()

        # --- Event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                env.reset()

        clock.tick(GameEnv.FPS)

    env.close()