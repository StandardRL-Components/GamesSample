
# Generated: 2025-08-28T05:55:41.815963
# Source Brief: brief_05732.md
# Brief Index: 5732

        
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
        "Controls: Arrows to move. When near a door, Shift cycles digits (0-9) and Space enters a digit."
    )

    game_description = (
        "An isometric puzzle game. Navigate the maze, find clues on the walls, and crack the 4-digit codes on locked doors to escape. You have 100 moves."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.MAZE_SIZE = (21, 21)
        self.MAX_MOVES = 100
        self.MAX_STEPS = 1000
        self.TILE_WIDTH, self.TILE_HEIGHT = 40, 20
        self.TILE_DEPTH = 20

        # Colors
        self.COLOR_BG = (15, 20, 30)
        self.COLOR_FLOOR = (40, 50, 60)
        self.COLOR_WALL_TOP = (80, 90, 110)
        self.COLOR_WALL_SIDE = (60, 70, 90)
        self.COLOR_PLAYER = (0, 200, 255)
        self.COLOR_PLAYER_GLOW = (0, 150, 200)
        self.COLOR_DOOR_LOCKED = (180, 50, 50)
        self.COLOR_DOOR_UNLOCKED = (50, 180, 50)
        self.COLOR_EXIT = (255, 215, 0)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_CORRECT = (0, 255, 100)
        self.COLOR_INCORRECT = (255, 50, 50)
        self.COLOR_UI_BG = (30, 40, 55, 200)

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_clue = pygame.font.SysFont("Consolas", 16, bold=True)
        self.font_code = pygame.font.SysFont("Consolas", 24, bold=True)

        # Game state variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.moves_remaining = 0
        self.maze = None
        self.player_pos = None
        self.start_pos = None
        self.exit_pos = None
        self.doors = {}
        self.rooms = []
        self.interaction_target = None
        self.selected_digit = 0
        self.current_code_attempt = ""
        self.code_feedback = []

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.moves_remaining = self.MAX_MOVES

        self._generate_maze()
        self.player_pos = self.start_pos

        self.interaction_target = None
        self.selected_digit = 0
        self.current_code_attempt = ""
        self.code_feedback = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        action_taken = any([movement > 0, space_held, shift_held])

        if action_taken:
            self.moves_remaining -= 1

        self._update_interaction_target()

        if self.interaction_target: # Code input mode
            if shift_held:
                self.selected_digit = (self.selected_digit + 1) % 10
                # pygame.mixer.Sound.play(sfx_cycle)
            if space_held:
                door_data = self.doors[self.interaction_target]
                slot_idx = len(self.current_code_attempt)
                if slot_idx < 4:
                    self.current_code_attempt += str(self.selected_digit)
                    correct_digit = door_data['code'][slot_idx]
                    if str(self.selected_digit) == correct_digit:
                        reward += 0.1
                        self.code_feedback.append(True)
                        # pygame.mixer.Sound.play(sfx_correct_digit)
                    else:
                        reward -= 0.1
                        self.code_feedback.append(False)
                        # pygame.mixer.Sound.play(sfx_incorrect_digit)

                if len(self.current_code_attempt) == 4:
                    if self.current_code_attempt == door_data['code']:
                        door_data['unlocked'] = True
                        reward += 5
                        self.interaction_target = None # Exit code mode
                        # pygame.mixer.Sound.play(sfx_door_unlock)
                    else:
                        # pygame.mixer.Sound.play(sfx_code_fail)
                        pass # Let player see feedback
                    self.current_code_attempt = ""
                    self.code_feedback = []

        else: # Movement mode
            if movement > 0:
                dx, dy = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)][movement]
                next_pos = (self.player_pos[0] + dx, self.player_pos[1] + dy)
                
                if 0 <= next_pos[0] < self.MAZE_SIZE[0] and 0 <= next_pos[1] < self.MAZE_SIZE[1]:
                    tile_type = self.maze[next_pos[1], next_pos[0]]
                    is_locked_door = next_pos in self.doors and not self.doors[next_pos]['unlocked']
                    if tile_type != 1 and not is_locked_door:
                        self.player_pos = next_pos
                        # pygame.mixer.Sound.play(sfx_move)

        self.score += reward
        terminated = self._check_termination()
        if terminated and self.moves_remaining <= 0:
            reward -= 100
            self.score -= 100
        if terminated and self.player_pos == self.exit_pos:
            reward += 100
            self.score += 100

        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            terminated = True
        
        if terminated:
            self.game_over = True

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _generate_maze(self):
        self.maze = np.ones(self.MAZE_SIZE, dtype=np.int8)
        self.doors = {}
        self.rooms = []

        # Define 4 rooms and corridors
        room_defs = [
            (1, 1, 8, 8),   # Top-left
            (11, 1, 18, 8), # Top-right
            (1, 11, 8, 18), # Bottom-left
            (11, 11, 18, 18) # Bottom-right
        ]
        for x1, y1, x2, y2 in room_defs:
            self.maze[y1:y2+1, x1:x2+1] = 0
        
        # Carve corridors
        self.maze[4:6, 1:19] = 0
        self.maze[1:19, 9:11] = 0
        self.maze[14:16, 1:19] = 0
        
        # Place doors
        door_positions = [(9, 5), (15, 9), (9, 15), (5, 9)]
        room_indices = [0, 1, 2, 3]
        self.np_random.shuffle(room_indices)
        
        for i, pos in enumerate(door_positions):
            self.maze[pos[1], pos[0]] = 2
            code = "".join([str(self.np_random.integers(0, 10)) for _ in range(4)])
            self.doors[pos] = {'code': code, 'unlocked': False, 'room_id': room_indices[i]}

        # Place start and exit
        self.start_pos = (4, 4)
        self.exit_pos = (15, 15)
        self.maze[self.start_pos[1], self.start_pos[0]] = 3
        self.maze[self.exit_pos[1], self.exit_pos[0]] = 4

        # Place clues
        for door_pos, door_data in self.doors.items():
            room_idx = door_data['room_id']
            x1, y1, x2, y2 = room_defs[room_idx]
            
            clue_wall_positions = []
            for y in range(y1, y2 + 1):
                clue_wall_positions.append((x1 - 1, y))
                clue_wall_positions.append((x2 + 1, y))
            for x in range(x1, x2 + 1):
                clue_wall_positions.append((x, y1 - 1))
                clue_wall_positions.append((x, y2 + 1))
            
            clue_positions = self.np_random.choice(len(clue_wall_positions), 4, replace=False)
            room_clues = []
            for i, digit in enumerate(door_data['code']):
                pos = clue_wall_positions[clue_positions[i]]
                room_clues.append({'pos': pos, 'digit': digit})
            self.rooms.append({'bounds': (x1,y1,x2,y2), 'clues': room_clues})

    def _update_interaction_target(self):
        # If already interacting, don't change target unless door is unlocked
        if self.interaction_target and self.interaction_target in self.doors and self.doors[self.interaction_target]['unlocked']:
            self.interaction_target = None
            self.current_code_attempt = ""
            self.code_feedback = []
            return

        # If already interacting, stay in that mode
        if self.interaction_target:
            return

        # Check for new interaction targets
        px, py = self.player_pos
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            check_pos = (px + dx, py + dy)
            if check_pos in self.doors and not self.doors[check_pos]['unlocked']:
                self.interaction_target = check_pos
                self.current_code_attempt = ""
                self.code_feedback = []
                self.selected_digit = 0
                return
        self.interaction_target = None # No adjacent locked doors

    def _check_termination(self):
        return self.moves_remaining <= 0 or self.player_pos == self.exit_pos

    def _grid_to_iso(self, x, y, z=0):
        iso_x = (x - y) * self.TILE_WIDTH / 2
        iso_y = (x + y) * self.TILE_HEIGHT / 2 - z * self.TILE_DEPTH
        return int(iso_x), int(iso_y)

    def _draw_iso_cube(self, surface, x, y, color_top, color_side, origin):
        ox, oy = origin
        
        # Calculate screen coordinates for the 8 corners of the cube
        p = [
            self._grid_to_iso(x, y, 0), self._grid_to_iso(x + 1, y, 0),
            self._grid_to_iso(x + 1, y + 1, 0), self._grid_to_iso(x, y + 1, 0),
            self._grid_to_iso(x, y, 1), self._grid_to_iso(x + 1, y, 1),
            self._grid_to_iso(x + 1, y + 1, 1), self._grid_to_iso(x, y + 1, 1)
        ]
        p = [(px + ox, py + oy) for px, py in p]

        # Draw top face
        pygame.gfxdraw.filled_polygon(surface, [p[4], p[5], p[6], p[7]], color_top)
        pygame.gfxdraw.aapolygon(surface, [p[4], p[5], p[6], p[7]], color_top)
        
        # Draw left face
        pygame.gfxdraw.filled_polygon(surface, [p[0], p[3], p[7], p[4]], color_side)
        pygame.gfxdraw.aapolygon(surface, [p[0], p[3], p[7], p[4]], color_side)

        # Draw right face
        pygame.gfxdraw.filled_polygon(surface, [p[3], p[2], p[6], p[7]], color_side)
        pygame.gfxdraw.aapolygon(surface, [p[3], p[2], p[6], p[7]], color_side)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        cam_center_x, cam_center_y = self.player_pos
        origin_x = self.SCREEN_WIDTH / 2
        origin_y = self.SCREEN_HEIGHT / 2 - self.TILE_HEIGHT
        
        # Adjust origin to keep player centered
        player_iso_x, player_iso_y = self._grid_to_iso(cam_center_x, cam_center_y)
        origin_x -= player_iso_x
        origin_y -= player_iso_y
        origin = (origin_x, origin_y)

        view_radius = 12
        min_y = max(0, cam_center_y - view_radius)
        max_y = min(self.MAZE_SIZE[1], cam_center_y + view_radius)
        min_x = max(0, cam_center_x - view_radius)
        max_x = min(self.MAZE_SIZE[0], cam_center_x + view_radius)

        # Render maze elements
        for y in range(min_y, max_y):
            for x in range(min_x, max_x):
                tile = self.maze[y, x]
                iso_x, iso_y = self._grid_to_iso(x, y)
                
                # Draw floor tile
                floor_points = [
                    self._grid_to_iso(x, y), self._grid_to_iso(x + 1, y),
                    self._grid_to_iso(x + 1, y + 1), self._grid_to_iso(x, y + 1)
                ]
                floor_points = [(px + origin[0], py + origin[1]) for px, py in floor_points]
                pygame.gfxdraw.filled_polygon(self.screen, floor_points, self.COLOR_FLOOR)

                if tile == 1: # Wall
                    self._draw_iso_cube(self.screen, x, y, self.COLOR_WALL_TOP, self.COLOR_WALL_SIDE, origin)
                elif (x, y) in self.doors: # Door
                    color = self.COLOR_DOOR_UNLOCKED if self.doors[(x, y)]['unlocked'] else self.COLOR_DOOR_LOCKED
                    self._draw_iso_cube(self.screen, x, y, color, tuple(c*0.8 for c in color), origin)
                elif tile == 4: # Exit
                    self._draw_iso_cube(self.screen, x, y, self.COLOR_EXIT, tuple(c*0.8 for c in self.COLOR_EXIT), origin)

        # Render clues
        for room in self.rooms:
            for clue in room['clues']:
                cx, cy = clue['pos']
                if min_x <= cx < max_x and min_y <= cy < max_y:
                    text_surf = self.font_clue.render(clue['digit'], True, self.COLOR_TEXT)
                    iso_x, iso_y = self._grid_to_iso(cx + 0.5, cy + 0.5, 0.5)
                    text_rect = text_surf.get_rect(center=(iso_x + origin[0], iso_y + origin[1]))
                    self.screen.blit(text_surf, text_rect)

        # Render player
        px, py = self.player_pos
        base_x, base_y = self._grid_to_iso(px, py, 0)
        base_y += self.TILE_HEIGHT / 2 # Center on tile
        
        # Glow effect
        for i in range(5, 0, -1):
            glow_color = list(self.COLOR_PLAYER_GLOW) + [50 - i * 8]
            radius = self.TILE_WIDTH / 4 + i * 2
            pygame.gfxdraw.filled_circle(self.screen, int(base_x + origin[0]), int(base_y + origin[1]), int(radius), glow_color)
        
        # Player cube
        player_color = self.COLOR_PLAYER
        pygame.draw.circle(self.screen, player_color, (int(base_x + origin[0]), int(base_y + origin[1])), int(self.TILE_WIDTH / 4))

    def _render_ui(self):
        # Moves remaining
        moves_text = f"Moves: {self.moves_remaining}"
        moves_surf = self.font_ui.render(moves_text, True, self.COLOR_TEXT)
        self.screen.blit(moves_surf, (10, 10))

        # Score
        score_text = f"Score: {self.score}"
        score_surf = self.font_ui.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (10, 35))

        # Code input UI
        if self.interaction_target:
            panel_w, panel_h = 280, 100
            panel_x, panel_y = (self.SCREEN_WIDTH - panel_w) / 2, self.SCREEN_HEIGHT - panel_h - 20
            
            s = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
            s.fill(self.COLOR_UI_BG)
            
            # Draw code slots
            code_str = self.current_code_attempt.ljust(4, '_')
            for i, char in enumerate(code_str):
                color = self.COLOR_TEXT
                if i < len(self.code_feedback):
                    color = self.COLOR_CORRECT if self.code_feedback[i] else self.COLOR_INCORRECT
                
                char_surf = self.font_code.render(char, True, color)
                char_rect = char_surf.get_rect(center=(40 + i * 40, 60))
                s.blit(char_surf, char_rect)

            # Draw digit selector
            selector_text = f"Select: {self.selected_digit}"
            selector_surf = self.font_ui.render(selector_text, True, self.COLOR_TEXT)
            s.blit(selector_surf, (190, 50))
            
            # Draw instructions
            instr_text = "Shift: Cycle, Space: Enter"
            instr_surf = self.font_ui.render(instr_text, True, self.COLOR_TEXT, None)
            instr_rect = instr_surf.get_rect(center=(panel_w / 2, 20))
            s.blit(instr_surf, instr_rect)
            
            self.screen.blit(s, (panel_x, panel_y))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_remaining": self.moves_remaining,
            "player_pos": self.player_pos,
        }

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

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    
    # Setup Pygame for display
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Isometric Code Breaker")
    clock = pygame.time.Clock()

    print(GameEnv.game_description)
    print(GameEnv.user_guide)

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
        
        # Only step if an action is taken in this turn-based game
        if any(action):
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']}, Moves Left: {info['moves_remaining']}")

        # Render the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(10) # Limit speed for manual play

    print(f"Game Over! Final Score: {env.score}")
    env.close()