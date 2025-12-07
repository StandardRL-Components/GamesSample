import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T18:02:36.064427
# Source Brief: brief_02531.md
# Brief Index: 2531
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "A tactical puzzle game where you must push blocks onto matching switches to open the exit, "
        "all while avoiding the watchful eyes of patrolling guards."
    )
    user_guide = (
        "Use arrow keys (↑↓←→) to move the cursor or push a selected block. "
        "Press space to select or deselect a block."
    )
    auto_advance = True

    # --- Constants ---
    COLOR_BG = (10, 10, 25)
    COLOR_GRID = (30, 30, 80)
    COLOR_WALL = (100, 100, 120)
    COLOR_CURSOR = (255, 255, 255)
    COLOR_TEXT = (220, 220, 220)
    COLOR_TEXT_SHADOW = (20, 20, 20)
    
    COLORS = {
        "red": (255, 50, 50),
        "green": (50, 255, 50),
        "blue": (50, 100, 255),
        "grey": (150, 150, 150)
    }
    
    COLOR_GUARD = (255, 150, 0)
    COLOR_GUARD_DETECTION = (255, 150, 0, 50) # with alpha
    COLOR_EXIT_CLOSED = (200, 200, 0)
    COLOR_EXIT_OPEN = (255, 255, 200)
    
    MAX_STEPS = 1000

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.width, self.height = 640, 400
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_msg = pygame.font.SysFont("monospace", 40, bold=True)

        # Game state variables
        self.level = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.victory = False
        self.grid_size = (0,0)
        self.cell_size = 0
        self.grid_offset = (0,0)
        self.cursor_pos = (0,0)
        self.blocks = []
        self.switches = []
        self.walls = []
        self.guards = []
        self.exit_pos = (0,0)
        self.exit_open = False
        self.selected_block_idx = None
        self.prev_space_held = False
        

    def _generate_level(self):
        self.level += 1
        
        # --- Level Progression ---
        w = 5 + (self.level - 1) // 5
        h = 5 + (self.level - 1) // 5
        self.grid_size = (min(w, 20), min(h, 14))
        
        num_colors = min(3, 1 + self.level // 3)
        self.active_colors = random.sample(list(self.COLORS.keys())[:3], num_colors)
        
        num_blocks_per_color = 1 + self.level // 6
        num_neutral_blocks = self.level // 4
        num_guards = self.level // 10
        
        # --- Grid Calculation ---
        self.cell_size = int(min( (self.width - 40) / self.grid_size[0], (self.height - 80) / self.grid_size[1] ))
        grid_w_px = self.grid_size[0] * self.cell_size
        grid_h_px = self.grid_size[1] * self.cell_size
        self.grid_offset = ((self.width - grid_w_px) // 2, (self.height - grid_h_px) // 2 + 20)

        # --- Generate Layout ---
        self.walls = []
        for x in range(-1, self.grid_size[0] + 1):
            self.walls.append((x, -1))
            self.walls.append((x, self.grid_size[1]))
        for y in range(self.grid_size[1]):
            self.walls.append((-1, y))
            self.walls.append((self.grid_size[0], y))
        
        # Add some internal walls
        for _ in range(self.grid_size[0] * self.grid_size[1] // 10):
            wall_pos = (random.randint(0, self.grid_size[0]-1), random.randint(0, self.grid_size[1]-1))
            if wall_pos not in self.walls:
                self.walls.append(wall_pos)

        # --- Place Entities ---
        possible_coords = [(x, y) for x in range(self.grid_size[0]) for y in range(self.grid_size[1]) if (x,y) not in self.walls]
        random.shuffle(possible_coords)
        
        self.exit_pos = (random.randint(0, self.grid_size[0]-1), self.grid_size[1])
        
        self.switches, self.blocks = [], []
        
        for color in self.active_colors:
            for _ in range(num_blocks_per_color):
                if len(possible_coords) < 2: break
                switch_pos = possible_coords.pop()
                block_pos = possible_coords.pop()
                self.switches.append({"pos": switch_pos, "color": color, "active": False})
                self.blocks.append({"pos": block_pos, "color": color})

        for _ in range(num_neutral_blocks):
            if not possible_coords: break
            self.blocks.append({"pos": possible_coords.pop(), "color": "grey"})

        self.guards = []
        for _ in range(num_guards):
            if not possible_coords: break
            start_pos = possible_coords.pop()
            path_len = random.randint(2, 5)
            path = [start_pos]
            curr = start_pos
            for i in range(path_len - 1):
                next_moves = []
                for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                    n_pos = (curr[0]+dx, curr[1]+dy)
                    if 0 <= n_pos[0] < self.grid_size[0] and 0 <= n_pos[1] < self.grid_size[1] and n_pos not in self.walls:
                        next_moves.append(n_pos)
                if not next_moves: break
                curr = random.choice(next_moves)
                path.append(curr)
            
            if len(path) > 1:
                self.guards.append({"path": path, "path_idx": 0, "direction": 1})

        self.cursor_pos = (self.grid_size[0] // 2, self.grid_size[1] // 2)
        if self.cursor_pos in self.walls:
             self.cursor_pos = possible_coords.pop() if possible_coords else (0,0)


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if options and "level" in options:
            self.level = options["level"] -1
        else:
            self.level = 0
        
        self._generate_level()
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.victory = False
        self.exit_open = False
        self.selected_block_idx = None
        self.prev_space_held = True # prevent action on first frame

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.prev_space_held
        
        reward = -0.01 # Cost of time
        self.steps += 1
        
        # --- 1. Handle Player Action ---
        threatened_before = {i: self._is_switch_threatened(s) for i, s in enumerate(self.switches)}

        if space_pressed:
            # sound: select_deselect.wav
            if self.selected_block_idx is not None:
                self.selected_block_idx = None
            else:
                for i, block in enumerate(self.blocks):
                    if block['pos'] == self.cursor_pos:
                        self.selected_block_idx = i
                        break
        
        # Move cursor or push block
        if movement != 0:
            dx, dy = [(0,0), (0,-1), (0,1), (-1,0), (1,0)][movement]
            
            if self.selected_block_idx is not None:
                # Push block
                block = self.blocks[self.selected_block_idx]
                target_pos = (block['pos'][0] + dx, block['pos'][1] + dy)
                
                is_wall = target_pos in self.walls
                is_occupied = any(b['pos'] == target_pos for b in self.blocks)
                
                if not is_wall and not is_occupied:
                    # sound: push_block.wav
                    block['pos'] = target_pos
                    self.cursor_pos = target_pos # Cursor follows pushed block
            else:
                # Move cursor
                target_pos = (self.cursor_pos[0] + dx, self.cursor_pos[1] + dy)
                if 0 <= target_pos[0] < self.grid_size[0] and 0 <= target_pos[1] < self.grid_size[1]:
                    self.cursor_pos = target_pos

        self.prev_space_held = space_held

        # --- 2. Update Game State & Calculate Immediate Rewards ---
        
        # Check switches
        for i, switch in enumerate(self.switches):
            was_active = switch['active']
            is_active = False
            for block in self.blocks:
                if block['pos'] == switch['pos'] and block['color'] == switch['color']:
                    is_active = True
                    break
            switch['active'] = is_active
            if is_active and not was_active:
                # sound: switch_on.wav
                reward += 0.1
        
        # Check if a block move exposed a switch
        threatened_after = {i: self._is_switch_threatened(s) for i, s in enumerate(self.switches)}
        for i in range(len(self.switches)):
            if not threatened_before[i] and threatened_after[i] and not self.switches[i]['active']:
                reward -= 0.1

        # Check for door opening
        was_exit_open = self.exit_open
        self.exit_open = all(s['active'] for s in self.switches) if self.switches else False
        if self.exit_open and not was_exit_open:
            # sound: door_open.wav
            reward += 5.0
            
        # --- 3. Move Guards ---
        for guard in self.guards:
            if not guard["path"]: continue
            
            path_idx = guard["path_idx"]
            direction = guard["direction"]
            
            path_idx += direction
            if not (0 <= path_idx < len(guard["path"])):
                guard["direction"] *= -1
                path_idx += 2 * guard["direction"]
            
            guard["path_idx"] = path_idx

        # --- 4. Check for Termination Conditions ---
        
        # Alarm condition
        for switch in self.switches:
            if not switch['active'] and self._is_switch_threatened(switch):
                # sound: alarm.wav
                self.game_over = True
                reward -= 10.0
                break
        
        # Victory condition
        if self.exit_open and self.selected_block_idx is not None and self.blocks[self.selected_block_idx]['pos'] == self.exit_pos:
            # sound: victory.wav
            self.game_over = True
            self.victory = True
            reward += 100.0
            
        # Max steps
        if self.steps >= self.MAX_STEPS:
            self.game_over = True

        self.score += reward
        terminated = self.game_over
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _is_switch_threatened(self, switch):
        for guard in self.guards:
            guard_pos = guard['path'][guard['path_idx']]
            dist = abs(guard_pos[0] - switch['pos'][0]) + abs(guard_pos[1] - switch['pos'][1])
            if dist <= 3:
                return True
        return False

    def _grid_to_pixel(self, x, y):
        px = self.grid_offset[0] + x * self.cell_size
        py = self.grid_offset[1] + y * self.cell_size
        return int(px), int(py)
    
    def _render_game(self):
        # Grid lines
        for x in range(self.grid_size[0] + 1):
            start = self._grid_to_pixel(x, 0)
            end = self._grid_to_pixel(x, self.grid_size[1])
            pygame.draw.line(self.screen, self.COLOR_GRID, start, (end[0], end[1] - 1), 1)
        for y in range(self.grid_size[1] + 1):
            start = self._grid_to_pixel(0, y)
            end = self._grid_to_pixel(self.grid_size[0], y)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, (end[0] - 1, end[1]), 1)

        # Walls
        for wx, wy in self.walls:
            if 0 <= wx < self.grid_size[0] and 0 <= wy < self.grid_size[1]:
                px, py = self._grid_to_pixel(wx, wy)
                pygame.draw.rect(self.screen, self.COLOR_WALL, (px, py, self.cell_size, self.cell_size))
                pygame.draw.rect(self.screen, self.COLOR_GRID, (px, py, self.cell_size, self.cell_size), 1)

        # Exit
        ex, ey = self._grid_to_pixel(self.exit_pos[0], self.exit_pos[1])
        exit_rect = pygame.Rect(ex, ey, self.cell_size, self.cell_size // 2)
        exit_color = self.COLOR_EXIT_OPEN if self.exit_open else self.COLOR_EXIT_CLOSED
        pygame.draw.rect(self.screen, exit_color, exit_rect)
        pygame.draw.rect(self.screen, self.COLOR_TEXT, exit_rect, 2)
        if self.exit_open:
            glow_surface = pygame.Surface((self.cell_size*2, self.cell_size*2), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(glow_surface, self.cell_size, self.cell_size, self.cell_size, (*exit_color, 40))
            self.screen.blit(glow_surface, (exit_rect.centerx - self.cell_size, exit_rect.centery - self.cell_size), special_flags=pygame.BLEND_RGBA_ADD)


        # Switches
        for switch in self.switches:
            sx, sy = self._grid_to_pixel(switch['pos'][0], switch['pos'][1])
            color = self.COLORS[switch['color']]
            center = (sx + self.cell_size // 2, sy + self.cell_size // 2)
            radius = self.cell_size // 3
            
            desat_color = tuple(c // 2 for c in color)
            pygame.gfxdraw.filled_circle(self.screen, center[0], center[1], radius, desat_color)
            pygame.gfxdraw.aacircle(self.screen, center[0], center[1], radius, color)
            
            if switch['active']:
                pulse = (math.sin(self.steps * 0.3) + 1) / 2
                glow_radius = int(radius + pulse * 5)
                glow_alpha = int(80 + pulse * 40)
                glow_surf = pygame.Surface((glow_radius*2, glow_radius*2), pygame.SRCALPHA)
                pygame.gfxdraw.filled_circle(glow_surf, glow_radius, glow_radius, glow_radius, (*color, glow_alpha))
                self.screen.blit(glow_surf, (center[0] - glow_radius, center[1] - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)
                pygame.gfxdraw.filled_circle(self.screen, center[0], center[1], radius, color)


        # Blocks
        for i, block in enumerate(self.blocks):
            bx, by = self._grid_to_pixel(block['pos'][0], block['pos'][1])
            color = self.COLORS[block['color']]
            rect = pygame.Rect(bx + 4, by + 4, self.cell_size - 8, self.cell_size - 8)
            
            pygame.draw.rect(self.screen, tuple(int(c*0.7) for c in color), rect, border_radius=3)
            pygame.draw.rect(self.screen, color, rect.inflate(-4,-4), border_radius=3)
            
            if i == self.selected_block_idx:
                pygame.draw.rect(self.screen, self.COLOR_CURSOR, (bx, by, self.cell_size, self.cell_size), 3, border_radius=4)


        # Guards and their detection range
        for guard in self.guards:
            guard_pos = guard['path'][guard['path_idx']]
            gx, gy = self._grid_to_pixel(guard_pos[0], guard_pos[1])
            center = (gx + self.cell_size // 2, gy + self.cell_size // 2)
            
            # Draw detection range first
            pulse = (math.sin(self.steps * 0.1) + 1) / 2
            det_radius_px = int(3 * self.cell_size + pulse * 5)
            temp_surf = pygame.Surface((det_radius_px*2, det_radius_px*2), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(temp_surf, det_radius_px, det_radius_px, det_radius_px, self.COLOR_GUARD_DETECTION)
            self.screen.blit(temp_surf, (center[0]-det_radius_px, center[1]-det_radius_px), special_flags=pygame.BLEND_RGBA_ADD)

            # Draw guard
            size = self.cell_size // 2
            points = [
                (center[0], center[1] - size // 2),
                (center[0] + size // 2, center[1]),
                (center[0], center[1] + size // 2),
                (center[0] - size // 2, center[1])
            ]
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_GUARD)
            pygame.gfxdraw.aapolygon(self.screen, points, tuple(int(c*0.7) for c in self.COLOR_GUARD))


        # Cursor
        if self.selected_block_idx is None:
            cx, cy = self._grid_to_pixel(self.cursor_pos[0], self.cursor_pos[1])
            pulse = (math.sin(self.steps * 0.2) + 1) / 2
            alpha = int(100 + pulse * 100)
            color = (*self.COLOR_CURSOR, alpha)
            rect = pygame.Rect(cx, cy, self.cell_size, self.cell_size)
            
            temp_surf = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
            pygame.draw.rect(temp_surf, color, (0,0, self.cell_size, self.cell_size), 3, border_radius=4)
            self.screen.blit(temp_surf, (cx, cy))


    def _render_ui(self):
        # Helper to draw text with shadow
        def draw_text(text, font, color, pos, shadow_color):
            shadow_surf = font.render(text, True, shadow_color)
            self.screen.blit(shadow_surf, (pos[0]+2, pos[1]+2))
            text_surf = font.render(text, True, color)
            self.screen.blit(text_surf, pos)
            
        # Score and Level
        draw_text(f"SCORE: {self.score:.2f}", self.font_ui, self.COLOR_TEXT, (10, 10), self.COLOR_TEXT_SHADOW)
        draw_text(f"LEVEL: {self.level}", self.font_ui, self.COLOR_TEXT, (self.width - 150, 10), self.COLOR_TEXT_SHADOW)
        draw_text(f"MOVES: {self.steps}", self.font_ui, self.COLOR_TEXT, (self.width / 2 - 60, 10), self.COLOR_TEXT_SHADOW)

        # Game Over / Victory Message
        if self.game_over:
            overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg = "VICTORY!" if self.victory else "ALARM TRIGGERED"
            color = self.COLOR_EXIT_OPEN if self.victory else self.COLOR_GUARD
            
            msg_surf = self.font_msg.render(msg, True, color)
            msg_rect = msg_surf.get_rect(center=(self.width/2, self.height/2))
            self.screen.blit(msg_surf, msg_rect)


    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
            "victory": self.victory,
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()
        
    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test reset to get initial observation
        obs, info = self.reset()
        
        # Test observation space  
        assert obs.shape == (self.height, self.width, 3)
        assert obs.dtype == np.uint8
        
        # Test reset return types
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.height, self.width, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block is for human play and debugging.
    # It will not be run by the evaluation server.
    # We need to unset the dummy video driver to see the window.
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv()
    env.validate_implementation()
    obs, info = env.reset()
    
    # Create a window for human play
    pygame.display.set_caption("Block Pusher")
    pygame.display.set_mode((env.width, env.height))
    
    running = True
    terminated = False
    
    while running:
        if terminated:
            print(f"Episode finished. Final Info: {info}")
            pygame.time.wait(2000) # Pause for 2 seconds on win/loss
            obs, info = env.reset(options={"level": env.level if info["victory"] else env.level - 1})
            terminated = False

        action = [0, 0, 0] # no-op, released, released
        
        # Event handling for human control
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: action[0] = 1
                elif event.key == pygame.K_DOWN: action[0] = 2
                elif event.key == pygame.K_LEFT: action[0] = 3
                elif event.key == pygame.K_RIGHT: action[0] = 4
                elif event.key == pygame.K_r: # Reset current level
                    obs, info = env.reset(options={"level": env.level})
                    continue
                elif event.key == pygame.K_n: # Skip to next level
                    obs, info = env.reset(options={"level": env.level + 1})
                    continue

        # Get held keys for MultiDiscrete actions
        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1
        
        obs, reward, terminated, truncated, info = env.step(action)

        # Pygame display for human playing
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        pygame.display.get_surface().blit(surf, (0,0))
        pygame.display.flip()

        env.clock.tick(15) # Limit frame rate for human play

    env.close()