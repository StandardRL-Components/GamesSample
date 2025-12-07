
# Generated: 2025-08-28T00:56:52.782534
# Source Brief: brief_03952.md
# Brief Index: 3952

        
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

    user_guide = (
        "Controls: Arrow keys to move the white pusher. "
        "Spacebar to push an adjacent block in the direction you last moved."
    )

    game_description = (
        "A minimalist puzzle game. Push the colored blocks onto their matching target "
        "squares. Plan your moves carefully, as each push costs a move and you have a limited supply."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 20, 12
        self.TILE_SIZE = 32
        self.MAX_MOVES = 50
        self.MAX_STEPS = 1000
        self.PARTICLE_LIFESPAN = 15

        # --- Colors ---
        self.COLOR_BG = (34, 34, 34)
        self.COLOR_GRID = (68, 68, 68)
        self.COLOR_WALL = (102, 102, 102)
        self.COLOR_PUSHER = (255, 255, 255)
        self.BLOCK_COLORS = {
            'A': (255, 87, 87),   # Red
            'B': (87, 155, 255),  # Blue
            'C': (87, 255, 155),  # Green
            'D': (255, 255, 87),  # Yellow
        }
        self.TARGET_COLORS = {
            'a': (120, 40, 40),
            'b': (40, 70, 120),
            'c': (40, 120, 70),
            'd': (120, 120, 40),
        }

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
        self.FONT_UI = pygame.font.Font(None, 24)
        self.FONT_MSG = pygame.font.Font(None, 60)
        self.FONT_MSG_SMALL = pygame.font.Font(None, 30)

        # --- Game State ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.moves_left = 0
        self.current_level = 1
        self.max_level_reached = 1
        self.pusher_pos = (0, 0)
        self.last_pusher_dir = (0, -1)
        self.blocks = []
        self.walls = set()
        self.targets = {}
        self.particles = []
        self._level_data = self._define_levels()
        self.max_level = len(self._level_data)
        self.win_message = ""

        self.reset()
        self.validate_implementation()
    
    def _define_levels(self):
        return {
            1: [
                "WWWWWWWWWWWWWWWWWWWW",
                "W..................W",
                "W...P..............W",
                "W..................W",
                "W....A.............W",
                "W..................W",
                "W.........a........W",
                "W..................W",
                "W..................W",
                "W..................W",
                "W..................W",
                "WWWWWWWWWWWWWWWWWWWW",
            ],
            2: [
                "WWWWWWWWWWWWWWWWWWWW",
                "W..................W",
                "W.P................W",
                "W...A.......B......W",
                "W..................W",
                "W..................W",
                "W...b.......a......W",
                "W..................W",
                "W..................W",
                "W..................W",
                "W..................W",
                "WWWWWWWWWWWWWWWWWWWW",
            ],
            3: [
                "WWWWWWWWWWWWWWWWWWWW",
                "W.P.a..............W",
                "W...WWWWWWWWWWWWW..W",
                "W.A.W..............W",
                "W...W...B..........W",
                "W...WWWWWWWW.......W",
                "W.......b..W.......W",
                "W..........W.......W",
                "W..........W.......W",
                "W..................W",
                "W..................W",
                "WWWWWWWWWWWWWWWWWWWW",
            ],
            4: [
                "WWWWWWWWWWWWWWWWWWWW",
                "W........P.........W",
                "W.A.B..............W",
                "W.b.a..............W",
                "W..................W",
                "W...C..............W",
                "W...c..............W",
                "W..................W",
                "W..................W",
                "W..................W",
                "W..................W",
                "WWWWWWWWWWWWWWWWWWWW",
            ],
            5: [
                "WWWWWWWWWWWWWWWWWWWW",
                "W.P.a.b.c..........W",
                "WWWWWWWWWWWWWWWW...W",
                "W.A.B.C............W",
                "W...WWWWWWWWWWWWWW.W",
                "W.d.W..............W",
                "W...W...D..........W",
                "W...WWWWWWWWWWWWWW.W",
                "W..................W",
                "W..................W",
                "W..................W",
                "WWWWWWWWWWWWWWWWWWWW",
            ],
        }

    def _load_level(self, level_num):
        self.walls.clear()
        self.blocks.clear()
        self.targets.clear()
        
        level_layout = self._level_data[level_num]
        block_id_counter = 0
        for r, row_str in enumerate(level_layout):
            for c, char in enumerate(row_str):
                pos = (c, r)
                if char == 'W':
                    self.walls.add(pos)
                elif char == 'P':
                    self.pusher_pos = pos
                elif char in self.BLOCK_COLORS:
                    self.blocks.append({
                        'id': block_id_counter,
                        'pos': pos,
                        'color_key': char,
                        'on_target': False
                    })
                    block_id_counter += 1
                elif char in self.TARGET_COLORS:
                    self.targets[pos] = char.upper()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if self.game_over: # If resetting after a win, advance level
             if self.win_message == "LEVEL COMPLETE!":
                self.current_level = min(self.current_level + 1, self.max_level)
                self.max_level_reached = max(self.current_level, self.max_level_reached)

        self._load_level(self.current_level)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.moves_left = self.MAX_MOVES
        self.last_pusher_dir = (0, -1)
        self.particles = []
        self.win_message = ""
        self._update_block_target_status()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        reward = 0
        action_taken = False

        on_target_before = sum(1 for b in self.blocks if b['on_target'])

        # --- Handle Movement ---
        if movement > 0:
            action_taken = True
            dx, dy = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)][movement]
            self.last_pusher_dir = (dx, dy)
            
            next_pos = (self.pusher_pos[0] + dx, self.pusher_pos[1] + dy)
            
            is_wall = next_pos in self.walls
            is_block = any(b['pos'] == next_pos for b in self.blocks)
            
            if not is_wall and not is_block:
                self.pusher_pos = next_pos

        # --- Handle Push ---
        if space_held:
            action_taken = True
            push_dir = self.last_pusher_dir
            block_to_push_pos = (self.pusher_pos[0] + push_dir[0], self.pusher_pos[1] + push_dir[1])
            
            block_positions = {b['pos']: b for b in self.blocks}
            
            if block_to_push_pos in block_positions:
                # Find the chain of blocks to push
                chain = []
                current_pos = block_to_push_pos
                while current_pos in block_positions:
                    chain.append(block_positions[current_pos])
                    current_pos = (current_pos[0] + push_dir[0], current_pos[1] + push_dir[1])
                
                # Check if the chain is blocked
                is_blocked = current_pos in self.walls or current_pos in block_positions
                
                if not is_blocked:
                    # Move all blocks in the chain
                    for block in reversed(chain):
                        old_pos = block['pos']
                        new_pos = (old_pos[0] + push_dir[0], old_pos[1] + push_dir[1])
                        block['pos'] = new_pos
                        # SFX: Block slide
                        self._create_particles(old_pos, self.BLOCK_COLORS[block['color_key']])

        if action_taken:
            self.moves_left -= 1
            reward -= 0.1

        self._update_block_target_status()
        on_target_after = sum(1 for b in self.blocks if b['on_target'])
        
        reward += (on_target_after - on_target_before)

        self.score += reward
        self.steps += 1
        terminated = self._check_termination()
        
        if terminated:
            if self.win_message == "LEVEL COMPLETE!":
                reward += 50
            else: # Out of moves
                reward -= 50
            self.score += reward
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _update_block_target_status(self):
        for block in self.blocks:
            block['on_target'] = self.targets.get(block['pos']) == block['color_key']

    def _check_termination(self):
        if all(b['on_target'] for b in self.blocks) and self.blocks:
            self.game_over = True
            self.win_message = "LEVEL COMPLETE!"
            # SFX: Level Win
            return True
        if self.moves_left <= 0:
            self.game_over = True
            self.win_message = "OUT OF MOVES"
            # SFX: Level Lose
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            self.win_message = "MAX STEPS REACHED"
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        grid_offset_x = (self.WIDTH - self.GRID_WIDTH * self.TILE_SIZE) // 2
        grid_offset_y = (self.HEIGHT - self.GRID_HEIGHT * self.TILE_SIZE) // 2

        # --- Draw Grid Lines ---
        for x in range(self.GRID_WIDTH + 1):
            start = (grid_offset_x + x * self.TILE_SIZE, grid_offset_y)
            end = (grid_offset_x + x * self.TILE_SIZE, grid_offset_y + self.GRID_HEIGHT * self.TILE_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end)
        for y in range(self.GRID_HEIGHT + 1):
            start = (grid_offset_x, grid_offset_y + y * self.TILE_SIZE)
            end = (grid_offset_x + self.GRID_WIDTH * self.TILE_SIZE, grid_offset_y + y * self.TILE_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end)

        # --- Draw Targets and Walls ---
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                pos = (c, r)
                rect = pygame.Rect(
                    grid_offset_x + c * self.TILE_SIZE,
                    grid_offset_y + r * self.TILE_SIZE,
                    self.TILE_SIZE, self.TILE_SIZE
                )
                if pos in self.targets:
                    color = self.TARGET_COLORS[self.targets[pos].lower()]
                    pygame.draw.rect(self.screen, color, rect)
                if pos in self.walls:
                    pygame.draw.rect(self.screen, self.COLOR_WALL, rect)
        
        # --- Draw Particles ---
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] = (p['pos'][0] + p['vel'][0], p['pos'][1] + p['vel'][1])
            p['life'] -= 1
            alpha = int(255 * (p['life'] / self.PARTICLE_LIFESPAN))
            color = (*p['color'], alpha)
            s = pygame.Surface((p['size'], p['size']), pygame.SRCALPHA)
            s.fill(color)
            self.screen.blit(s, (int(p['pos'][0]), int(p['pos'][1])))

        # --- Draw Blocks ---
        for block in self.blocks:
            color = self.BLOCK_COLORS[block['color_key']]
            rect = pygame.Rect(
                grid_offset_x + block['pos'][0] * self.TILE_SIZE + 2,
                grid_offset_y + block['pos'][1] * self.TILE_SIZE + 2,
                self.TILE_SIZE - 4, self.TILE_SIZE - 4
            )
            pygame.draw.rect(self.screen, color, rect, border_radius=4)
            if block['on_target']:
                pygame.gfxdraw.aacircle(self.screen, rect.centerx, rect.centery, self.TILE_SIZE // 4, (255, 255, 255))
                pygame.gfxdraw.filled_circle(self.screen, rect.centerx, rect.centery, self.TILE_SIZE // 4, (255, 255, 255))

        # --- Draw Pusher ---
        pusher_rect = pygame.Rect(
            grid_offset_x + self.pusher_pos[0] * self.TILE_SIZE + 4,
            grid_offset_y + self.pusher_pos[1] * self.TILE_SIZE + 4,
            self.TILE_SIZE - 8, self.TILE_SIZE - 8
        )
        pygame.draw.rect(self.screen, self.COLOR_PUSHER, pusher_rect, border_radius=3)
        
        # Draw direction indicator
        center = pusher_rect.center
        dx, dy = self.last_pusher_dir
        p1 = (center[0] + dx * 6, center[1] + dy * 6)
        p2 = (center[0] - dy * 4, center[1] + dx * 4)
        p3 = (center[0] + dy * 4, center[1] - dx * 4)
        pygame.draw.polygon(self.screen, self.COLOR_BG, (p1, p2, p3))


    def _render_ui(self):
        # --- Moves Left ---
        moves_text = self.FONT_UI.render(f"Moves: {self.moves_left}", True, (255, 255, 255))
        self.screen.blit(moves_text, (10, 10))

        # --- Level ---
        level_text = self.FONT_UI.render(f"Level: {self.current_level}/{self.max_level}", True, (255, 255, 255))
        self.screen.blit(level_text, (self.WIDTH - level_text.get_width() - 10, 10))

        # --- Game Over Message ---
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg_text = self.FONT_MSG.render(self.win_message, True, (255, 255, 255))
            msg_rect = msg_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2 - 20))
            self.screen.blit(msg_text, msg_rect)

            reset_msg_text = self.FONT_MSG_SMALL.render("Call reset() to continue", True, (200, 200, 200))
            reset_rect = reset_msg_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2 + 30))
            self.screen.blit(reset_msg_text, reset_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.current_level,
            "moves_left": self.moves_left,
            "blocks_on_target": sum(1 for b in self.blocks if b['on_target']),
            "total_blocks": len(self.blocks),
        }

    def _create_particles(self, grid_pos, color):
        grid_offset_x = (self.WIDTH - self.GRID_WIDTH * self.TILE_SIZE) // 2
        grid_offset_y = (self.HEIGHT - self.GRID_HEIGHT * self.TILE_SIZE) // 2
        
        px = grid_offset_x + grid_pos[0] * self.TILE_SIZE + self.TILE_SIZE / 2
        py = grid_offset_y + grid_pos[1] * self.TILE_SIZE + self.TILE_SIZE / 2

        for _ in range(10):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 3)
            self.particles.append({
                'pos': [px, py],
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': self.PARTICLE_LIFESPAN,
                'color': color,
                'size': random.randint(3, 6)
            })

    def close(self):
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Game loop
    running = True
    # Use a separate screen for display if running manually
    display_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Pusher Puzzle")
    clock = pygame.time.Clock()

    action = env.action_space.sample()
    action.fill(0) # Start with no-op

    while running:
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                done = False

        if not done:
            # --- Get Player Input ---
            keys = pygame.key.get_pressed()
            action.fill(0)
            if keys[pygame.K_UP]:
                action[0] = 1
            elif keys[pygame.K_DOWN]:
                action[0] = 2
            elif keys[pygame.K_LEFT]:
                action[0] = 3
            elif keys[pygame.K_RIGHT]:
                action[0] = 4

            if keys[pygame.K_SPACE]:
                action[1] = 1
            
            # --- Step the Environment ---
            # Only step if an action is taken, because auto_advance is False
            if np.any(action):
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                print(f"Action: {action}, Reward: {reward:.2f}, Info: {info}")

        # --- Rendering ---
        # The environment's observation is already a rendered frame
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        display_screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) # Limit FPS for manual play

    env.close()