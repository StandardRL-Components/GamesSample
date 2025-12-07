
# Generated: 2025-08-27T13:25:29.230395
# Source Brief: brief_00364.md
# Brief Index: 364

        
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
        "Controls: Use arrow keys to move. Push the colored boxes into their matching zones."
    )

    game_description = (
        "A minimalist puzzle game. Push all boxes into their designated zones within 30 moves to win. Plan your moves carefully to avoid getting stuck!"
    )

    auto_advance = False
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_WIDTH = 16
    GRID_HEIGHT = 10
    TILE_SIZE = 40
    
    MAX_MOVES = 30
    MAX_STEPS = 1000

    # Colors
    COLOR_BG = (25, 25, 35)
    COLOR_GRID = (45, 45, 55)
    COLOR_WALL = (80, 80, 95)
    COLOR_PLAYER = (255, 255, 255)
    
    # Box and Zone Colors
    COLORS = {
        "red": (255, 80, 80),
        "green": (80, 255, 80),
        "blue": (80, 120, 255),
        "yellow": (255, 220, 80),
    }
    ZONE_ALPHA = 100

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        self.font_ui = pygame.font.Font(None, 28)
        self.font_msg = pygame.font.Font(None, 64)
        
        # Game state variables are initialized in reset()
        self.player_pos = None
        self.player_dir = None
        self.boxes = None
        self.zones = None
        self.walls = None
        self.steps = 0
        self.score = 0
        self.moves_left = 0
        self.game_over = False
        self.win = False
        self.last_reward_info = ""

        # This will be populated in reset()
        self.level_layout = self._get_level_layout()
        
        self.reset()
        self.validate_implementation()
    
    def _get_level_layout(self):
        # A static, solvable level layout
        layout = [
            "WWWWWWWWWWWWWWWW",
            "W..............W",
            "W.r.R.WWWWW.g..W",
            "W.....W...P.G..W",
            "W.y...W.B......W",
            "W.Y...WWWWW....W",
            "W..............W",
            "W.......b......W",
            "W..............W",
            "WWWWWWWWWWWWWWWW",
        ]
        return layout

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.moves_left = self.MAX_MOVES
        self.game_over = False
        self.win = False
        self.last_reward_info = ""
        self.player_dir = (0, 1) # Start facing down

        self.boxes = []
        self.zones = []
        self.walls = set()
        
        color_keys = list(self.COLORS.keys())
        box_map = {'R': 'red', 'G': 'green', 'B': 'blue', 'Y': 'yellow'}
        zone_map = {'r': 'red', 'g': 'green', 'b': 'blue', 'y': 'yellow'}

        for y, row in enumerate(self.level_layout):
            for x, char in enumerate(row):
                pos = (x, y)
                if char == 'W':
                    self.walls.add(pos)
                elif char == 'P':
                    self.player_pos = pos
                elif char in box_map:
                    color_name = box_map[char]
                    self.boxes.append({
                        "id": color_name,
                        "pos": pos,
                        "color": self.COLORS[color_name]
                    })
                elif char in zone_map:
                    color_name = zone_map[char]
                    self.zones.append({
                        "id": color_name,
                        "pos": pos,
                        "color": self.COLORS[color_name]
                    })
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = 0
        
        # Only process non-noop actions
        if movement > 0:
            self.moves_left -= 1
            reward = -0.1
            self.last_reward_info = "-0.1 (Move)"

            # 1=up, 2=down, 3=left, 4=right
            move_map = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
            dx, dy = move_map[movement]
            self.player_dir = (dx, dy)

            px, py = self.player_pos
            next_pos = (px + dx, py + dy)

            if next_pos not in self.walls:
                box_at_next = self._get_box_at(next_pos)
                
                if box_at_next:
                    # Trying to push a box
                    box_next_pos = (next_pos[0] + dx, next_pos[1] + dy)
                    
                    is_blocked = (box_next_pos in self.walls) or (self._get_box_at(box_next_pos) is not None)
                    
                    if not is_blocked:
                        # --- Push successful ---
                        # Move box
                        box_at_next['pos'] = box_next_pos
                        # Move player
                        self.player_pos = next_pos
                        
                        # Check for rewards related to the moved box
                        is_on_correct_zone = self._is_on_correct_zone(box_at_next)
                        if is_on_correct_zone:
                            reward += 1.0
                            self.last_reward_info += " | +1.0 (Zone)"
                            # Sound: sfx_box_in_zone.wav
                        
                        if self._is_stuck(box_at_next):
                            reward -= 2.0
                            self.last_reward_info += " | -2.0 (Stuck)"
                            # Sound: sfx_box_stuck.wav
                        else:
                            # Sound: sfx_box_push.wav
                            pass

                else:
                    # --- Move into empty space ---
                    self.player_pos = next_pos
                    # Sound: sfx_player_move.wav
        
        self.score += reward
        
        # Check termination conditions
        if self._check_win_condition():
            self.win = True
            self.game_over = True
            win_reward = 100.0
            self.score += win_reward
            reward += win_reward
            self.last_reward_info = "+100 (WIN!)"
            # Sound: sfx_win.wav
        elif self.moves_left <= 0:
            self.game_over = True
            loss_penalty = -10.0
            self.score += loss_penalty
            reward += loss_penalty
            self.last_reward_info = "-10 (Out of moves)"
            # Sound: sfx_lose.wav

        if self.steps >= self.MAX_STEPS:
            self.game_over = True

        self.steps += 1
        terminated = self.game_over
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _get_box_at(self, pos):
        for box in self.boxes:
            if box['pos'] == pos:
                return box
        return None

    def _get_zone_at(self, pos):
        for zone in self.zones:
            if zone['pos'] == pos:
                return zone
        return None

    def _is_on_correct_zone(self, box):
        zone = self._get_zone_at(box['pos'])
        return zone is not None and zone['id'] == box['id']

    def _is_stuck(self, box):
        # A box is stuck if it's in a corner and not on its goal zone
        if self._is_on_correct_zone(box):
            return False

        x, y = box['pos']
        
        is_wall_or_box = lambda p: p in self.walls or (self._get_box_at(p) and self._get_box_at(p) != box)

        # Check for corners
        up_blocked = is_wall_or_box((x, y - 1))
        down_blocked = is_wall_or_box((x, y + 1))
        left_blocked = is_wall_or_box((x - 1, y))
        right_blocked = is_wall_or_box((x + 1, y))

        if (up_blocked and left_blocked) or \
           (up_blocked and right_blocked) or \
           (down_blocked and left_blocked) or \
           (down_blocked and right_blocked):
            return True
        return False

    def _check_win_condition(self):
        return all(self._is_on_correct_zone(box) for box in self.boxes)

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
            "moves_left": self.moves_left,
            "zones_filled": sum(1 for box in self.boxes if self._is_on_correct_zone(box))
        }

    def _render_game(self):
        # Draw grid
        for x in range(0, self.SCREEN_WIDTH, self.TILE_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.TILE_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

        # Draw zones (background elements)
        for zone in self.zones:
            zx, zy = zone['pos']
            center_x = int(zx * self.TILE_SIZE + self.TILE_SIZE / 2)
            center_y = int(zy * self.TILE_SIZE + self.TILE_SIZE / 2)
            radius = int(self.TILE_SIZE * 0.35)
            
            # Create a temporary surface for transparency
            temp_surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(temp_surf, radius, radius, radius, (*zone['color'], self.ZONE_ALPHA))
            pygame.gfxdraw.aacircle(temp_surf, radius, radius, radius, (*zone['color'], self.ZONE_ALPHA + 50))
            self.screen.blit(temp_surf, (center_x - radius, center_y - radius))

        # Draw walls
        for wx, wy in self.walls:
            rect = pygame.Rect(wx * self.TILE_SIZE, wy * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_WALL, rect)

        # Draw boxes
        for box in self.boxes:
            bx, by = box['pos']
            rect = pygame.Rect(
                bx * self.TILE_SIZE + self.TILE_SIZE * 0.1,
                by * self.TILE_SIZE + self.TILE_SIZE * 0.1,
                self.TILE_SIZE * 0.8,
                self.TILE_SIZE * 0.8
            )
            pygame.draw.rect(self.screen, box['color'], rect, border_radius=4)
            # Add a highlight for depth
            highlight_color = tuple(min(255, c + 40) for c in box['color'])
            pygame.draw.rect(self.screen, highlight_color, rect.inflate(-6, -6), border_radius=3)

        # Draw player
        px, py = self.player_pos
        center_x = px * self.TILE_SIZE + self.TILE_SIZE / 2
        center_y = py * self.TILE_SIZE + self.TILE_SIZE / 2
        
        p_size = self.TILE_SIZE * 0.3
        dx, dy = self.player_dir
        
        p1 = (center_x + dx * p_size, center_y + dy * p_size)
        p2 = (center_x - dy * p_size, center_y + dx * p_size)
        p3 = (center_x + dy * p_size, center_y - dx * p_size)
        
        points = [(int(p[0]), int(p[1])) for p in [p1, p2, p3]]
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

    def _render_ui(self):
        # Moves Left
        moves_text = self.font_ui.render(f"Moves: {self.moves_left}", True, (200, 200, 220))
        self.screen.blit(moves_text, (10, 10))

        # Zones Filled
        zones_filled = sum(1 for box in self.boxes if self._is_on_correct_zone(box))
        zones_text = self.font_ui.render(f"Zones: {zones_filled} / {len(self.zones)}", True, (200, 200, 220))
        self.screen.blit(zones_text, (self.SCREEN_WIDTH - zones_text.get_width() - 10, 10))
        
        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            msg_text_str = "YOU WIN!" if self.win else "GAME OVER"
            msg_color = self.COLORS['green'] if self.win else self.COLORS['red']
            msg_surf = self.font_msg.render(msg_text_str, True, msg_color)
            msg_rect = msg_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(msg_surf, msg_rect)

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Set up Pygame window for human play
    pygame.display.set_caption("Sokoban Puzzle")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    print(GameEnv.game_description)
    print(GameEnv.user_guide)

    running = True
    while running:
        # Default action is no-op
        action = [0, 0, 0] 
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
                elif event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
                elif event.key == pygame.K_r: # Reset button
                    obs, info = env.reset()
                    print("--- Game Reset ---")
                elif event.key == pygame.K_q:
                    running = False
        
        # Only step if a move was made
        if action[0] != 0:
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Action: {action}, Reward: {reward:.2f}, Terminated: {terminated}, Info: {info}")
            if terminated:
                print("Game Over. Press 'R' to restart or 'Q' to quit.")

        # Render the observation to the display window
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30)

    pygame.quit()