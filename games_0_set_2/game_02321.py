
# Generated: 2025-08-28T04:29:38.728101
# Source Brief: brief_02321.md
# Brief Index: 2321

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
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
        "Controls: Arrow keys to push the selected block. Hold Shift to cycle selection. Solve the puzzle before time runs out."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced puzzle game. Push colored blocks onto their matching targets against the clock. Select different blocks using Shift."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()
        
        # Constants
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.GRID_W, self.GRID_H = 16, 10
        self.TILE_SIZE = 40
        self.NUM_BLOCKS = 5
        self.MAX_STEPS = 90 * 30  # 90 seconds @ 30fps
        self.SHUFFLE_MOVES = 100

        # Colors
        self.COLOR_BG = (25, 28, 32)
        self.COLOR_GRID = (40, 45, 50)
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_SELECTOR = (255, 255, 0)
        self.BLOCK_COLORS = [
            (255, 80, 80),   # Red
            (80, 255, 80),   # Green
            (80, 150, 255),  # Blue
            (255, 255, 80),  # Yellow
            (255, 80, 255),  # Magenta
            (80, 255, 255),  # Cyan
        ]

        # Fonts
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_msg = pygame.font.SysFont("monospace", 48, bold=True)

        # State variables (initialized in reset)
        self.blocks = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.current_selection_idx = 0
        self.movable_block_indices = []
        self.prev_shift_held = False
        
        # Initialize state
        self.reset()
        self.validate_implementation()

    def _generate_puzzle(self):
        """Generates a solvable puzzle by starting with the solution and shuffling."""
        
        # 1. Generate unique target locations
        possible_coords = [(x, y) for x in range(self.GRID_W) for y in range(self.GRID_H)]
        target_coords = self.np_random.choice(len(possible_coords), self.NUM_BLOCKS, replace=False)
        target_coords = [possible_coords[i] for i in target_coords]

        # 2. Create blocks on their targets (solved state)
        blocks = []
        for i in range(self.NUM_BLOCKS):
            blocks.append({
                "pos": target_coords[i],
                "target_pos": target_coords[i],
                "color": self.BLOCK_COLORS[i % len(self.BLOCK_COLORS)],
                "solved": True,
            })

        # 3. Shuffle the blocks with random moves
        for _ in range(self.SHUFFLE_MOVES):
            block_idx = self.np_random.integers(0, self.NUM_BLOCKS)
            direction = self.np_random.integers(1, 5) # 1-4 for directions
            
            # Use a static helper for the push logic during generation
            GameEnv._static_push_logic(blocks, block_idx, direction, self.GRID_W, self.GRID_H)

        # 4. Finalize state for the start of the game
        is_solved = True
        for block in blocks:
            if block["pos"] == block["target_pos"]:
                block["solved"] = True
            else:
                block["solved"] = False
                is_solved = False
        
        # If by some miracle it shuffles back to the solved state, regenerate
        if is_solved:
            return self._generate_puzzle()
            
        return blocks

    @staticmethod
    def _static_push_logic(blocks, start_idx, direction, grid_w, grid_h):
        """A version of the push logic for puzzle generation that operates on a block list."""
        dirs = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
        dx, dy = dirs[direction]

        chain = []
        current_pos = blocks[start_idx]["pos"]
        
        while True:
            next_pos = (current_pos[0] + dx, current_pos[1] + dy)
            
            # Check boundaries
            if not (0 <= next_pos[0] < grid_w and 0 <= next_pos[1] < grid_h):
                return False # Push failed, hit a wall

            # Find block at next position
            next_block_idx = -1
            for i, b in enumerate(blocks):
                if b["pos"] == next_pos:
                    next_block_idx = i
                    break
            
            if next_block_idx != -1:
                chain.append(next_block_idx)
                current_pos = next_pos
            else:
                # Path is clear for the last block in the chain
                break
        
        # Move all blocks in the chain, starting from the end
        for i in reversed(chain):
            blocks[i]["pos"] = (blocks[i]["pos"][0] + dx, blocks[i]["pos"][1] + dy)
        
        # Move the starting block
        blocks[start_idx]["pos"] = (blocks[start_idx]["pos"][0] + dx, blocks[start_idx]["pos"][1] + dy)
        return True

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.particles.clear()
        self.prev_shift_held = False

        self.blocks = self._generate_puzzle()
        self._update_movable_blocks()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = -0.005  # Time penalty to encourage speed (calibrated for 30fps)

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Handle selection cycling on shift press (rising edge)
        if shift_held and not self.prev_shift_held and self.movable_block_indices:
            self.current_selection_idx = (self.current_selection_idx + 1) % len(self.movable_block_indices)
        self.prev_shift_held = shift_held

        # Handle push action
        if movement > 0 and self.movable_block_indices:
            block_to_move_real_idx = self.movable_block_indices[self.current_selection_idx]
            self._push_block(block_to_move_real_idx, movement)
            # sfx_push

        # Check for newly solved blocks
        newly_solved = False
        for i, block in enumerate(self.blocks):
            if not block["solved"] and block["pos"] == block["target_pos"]:
                block["solved"] = True
                reward += 1.0
                newly_solved = True
                self._spawn_solve_particles(block)
                # sfx_solve_block
        
        if newly_solved:
            self._update_movable_blocks()

        # Check for termination conditions
        terminated = False
        all_solved = not self.movable_block_indices
        
        if all_solved:
            terminated = True
            self.game_over = True
            reward += 50.0
            # sfx_win
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
            reward -= 50.0
            # sfx_lose

        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _push_block(self, start_block_idx, direction):
        """Attempts to push a block and any blocks in its path."""
        dirs = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)} # up, down, left, right
        dx, dy = dirs[direction]

        chain = [start_block_idx]
        current_pos = self.blocks[start_block_idx]["pos"]
        
        while True:
            next_pos = (current_pos[0] + dx, current_pos[1] + dy)
            
            if not (0 <= next_pos[0] < self.GRID_W and 0 <= next_pos[1] < self.GRID_H):
                return False # Hit wall

            found_block_idx = -1
            for i, b in enumerate(self.blocks):
                if b["pos"] == next_pos:
                    if b["solved"]:
                        return False # Blocked by a solved block
                    found_block_idx = i
                    break
            
            if found_block_idx != -1:
                chain.append(found_block_idx)
                current_pos = next_pos
            else:
                break # Path is clear
        
        # Move all blocks in chain, from end to start
        for i in reversed(chain):
            self.blocks[i]["pos"] = (self.blocks[i]["pos"][0] + dx, self.blocks[i]["pos"][1] + dy)
        
        return True

    def _update_movable_blocks(self):
        """Updates the list of indices for blocks that are not yet solved."""
        self.movable_block_indices = [i for i, b in enumerate(self.blocks) if not b["solved"]]
        if not self.movable_block_indices:
            self.current_selection_idx = 0
        elif self.current_selection_idx >= len(self.movable_block_indices):
            self.current_selection_idx = 0

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._draw_grid()
        self._draw_targets()
        self._update_and_draw_particles()
        self._draw_blocks()
        self._draw_selector()

        if self.game_over:
            self._draw_end_message()

    def _draw_grid(self):
        for x in range(0, self.SCREEN_WIDTH, self.TILE_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.TILE_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

    def _draw_targets(self):
        for block in self.blocks:
            gx, gy = block["target_pos"]
            cx = int((gx + 0.5) * self.TILE_SIZE)
            cy = int((gy + 0.5) * self.TILE_SIZE)
            radius = int(self.TILE_SIZE * 0.35)
            
            target_color = tuple(c * 0.5 for c in block["color"])
            pygame.gfxdraw.filled_circle(self.screen, cx, cy, radius, target_color)
            pygame.gfxdraw.aacircle(self.screen, cx, cy, radius, block["color"])

    def _draw_blocks(self):
        for block in self.blocks:
            gx, gy = block["pos"]
            rect = pygame.Rect(
                gx * self.TILE_SIZE + 2,
                gy * self.TILE_SIZE + 2,
                self.TILE_SIZE - 4,
                self.TILE_SIZE - 4
            )
            border_color = tuple(max(0, c - 50) for c in block["color"])
            pygame.draw.rect(self.screen, block["color"], rect, border_radius=4)
            pygame.draw.rect(self.screen, border_color, rect, width=2, border_radius=4)

    def _draw_selector(self):
        if not self.movable_block_indices or self.game_over:
            return

        selected_block_idx = self.movable_block_indices[self.current_selection_idx]
        block = self.blocks[selected_block_idx]
        gx, gy = block["pos"]
        
        # Pulsing effect for the selector
        pulse = (math.sin(self.steps * 0.2) + 1) / 2  # Varies between 0 and 1
        padding = int(2 + pulse * 3)
        
        rect = pygame.Rect(
            gx * self.TILE_SIZE + padding,
            gy * self.TILE_SIZE + padding,
            self.TILE_SIZE - padding * 2,
            self.TILE_SIZE - padding * 2
        )
        
        # Create a temporary surface for transparency
        selector_surf = pygame.Surface((self.TILE_SIZE, self.TILE_SIZE), pygame.SRCALPHA)
        alpha = int(100 + pulse * 100)
        color = self.COLOR_SELECTOR + (alpha,)
        pygame.draw.rect(selector_surf, color, selector_surf.get_rect(), width=3, border_radius=6)
        
        self.screen.blit(selector_surf, (gx * self.TILE_SIZE, gy * self.TILE_SIZE))

    def _spawn_solve_particles(self, solved_block):
        cx = (solved_block["pos"][0] + 0.5) * self.TILE_SIZE
        cy = (solved_block["pos"][1] + 0.5) * self.TILE_SIZE
        for _ in range(30):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                "pos": [cx, cy],
                "vel": vel,
                "color": solved_block["color"],
                "life": self.np_random.integers(20, 40)
            })

    def _update_and_draw_particles(self):
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][0] *= 0.95
            p["vel"][1] *= 0.95
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)
            else:
                radius = int(p["life"] / 8)
                if radius > 0:
                    alpha = int((p["life"] / 40) * 255)
                    color = p["color"] + (alpha,)
                    
                    # Create a temporary surface for the particle to handle alpha
                    particle_surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
                    pygame.draw.circle(particle_surf, color, (radius, radius), radius)
                    self.screen.blit(particle_surf, (int(p["pos"][0] - radius), int(p["pos"][1] - radius)))

    def _render_ui(self):
        time_left = max(0, (self.MAX_STEPS - self.steps) / 30)
        time_text = f"Time: {time_left:.1f}s"
        
        num_solved = self.NUM_BLOCKS - len(self.movable_block_indices)
        solved_text = f"Solved: {num_solved}/{self.NUM_BLOCKS}"
        
        time_surf = self.font_ui.render(time_text, True, self.COLOR_UI_TEXT)
        solved_surf = self.font_ui.render(solved_text, True, self.COLOR_UI_TEXT)
        
        self.screen.blit(time_surf, (self.SCREEN_WIDTH - time_surf.get_width() - 10, 10))
        self.screen.blit(solved_surf, (self.SCREEN_WIDTH - solved_surf.get_width() - 10, 30))

    def _draw_end_message(self):
        all_solved = not self.movable_block_indices
        message = "COMPLETE!" if all_solved else "TIME UP!"
        color = (100, 255, 100) if all_solved else (255, 100, 100)

        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))

        text_surf = self.font_msg.render(message, True, color)
        text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
        self.screen.blit(text_surf, text_rect)
        
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining_steps": self.MAX_STEPS - self.steps,
            "blocks_solved": self.NUM_BLOCKS - len(self.movable_block_indices),
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
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    import os
    # Set a non-display backend for server-side execution if needed
    # os.environ["SDL_VIDEODRIVER"] = "dummy"

    env = GameEnv(render_mode="rgb_array")
    
    # --- To play manually ---
    # This requires a display. Comment out the dummy driver line above.
    pygame.display.set_caption("Block Pusher")
    screen = pygame.display.set_mode((640, 400))
    
    obs, info = env.reset()
    terminated = False
    
    while not terminated:
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
        
        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)

        # Draw the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Run at 30 FPS

    print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
    
    # Keep the final screen visible for a moment
    pygame.time.wait(2000)
    env.close()