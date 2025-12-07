
# Generated: 2025-08-27T13:07:17.254732
# Source Brief: brief_00266.md
# Brief Index: 266

        
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
    metadata = {"render_modes": ["rgb_array"]}

    # Short, user-facing control string
    user_guide = (
        "Controls: Use arrow keys to move the cursor. "
        "Press Space to pick up or drop a colored ball. "
        "Sort all balls into their matching colored zones."
    )

    # Short, user-facing description of the game
    game_description = (
        "A minimalist color sorting puzzle. Drag and drop colored balls into their "
        "designated zones before you run out of moves. Each drop costs one move."
    )

    # Frames only advance when an action is received
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_COLS, GRID_ROWS = 16, 10
    CELL_SIZE = 40
    BALL_RADIUS = 16
    NUM_BALLS_PER_COLOR = 3
    TOTAL_MOVES = 30
    
    # --- Colors ---
    COLOR_BG = (25, 25, 35)
    COLOR_GRID = (40, 40, 50)
    COLOR_CURSOR = (255, 255, 255)
    COLOR_TEXT = (220, 220, 220)
    
    COLORS = {
        "red": (255, 80, 80),
        "green": (80, 255, 80),
        "blue": (80, 80, 255),
    }
    TARGET_COLORS = {
        "red": (60, 30, 30),
        "green": (30, 60, 30),
        "blue": (30, 30, 60),
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Arial", 24)
        self.font_big = pygame.font.SysFont("Arial", 48, bold=True)
        
        # Initialize state variables
        self.balls = []
        self.target_zones = {}
        self.grid = np.full((self.GRID_COLS, self.GRID_ROWS), -1, dtype=int)
        self.cursor_pos = np.array([self.GRID_COLS // 2, self.GRID_ROWS // 2])
        self.held_ball_id = None
        self.last_space_held = False
        self.moves_left = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.win_message = ""
        self.particles = []

        self.reset()
        
        # Validate implementation after initialization
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.moves_left = self.TOTAL_MOVES
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_message = ""
        self.held_ball_id = None
        self.last_space_held = False
        self.cursor_pos = np.array([self.GRID_COLS // 2, self.GRID_ROWS // 2])
        self.particles.clear()

        self._setup_zones_and_balls()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, _ = action
        reward = 0
        
        self._handle_input(movement, space_held)
        
        # A move is only consumed when a ball is dropped
        # The reward is also calculated upon dropping
        reward, move_made = self._process_drop(space_held)
        if move_made:
            self.score += reward

        self.steps += 1
        terminated = self._check_termination()
        
        if terminated and not self.game_over:
            self.game_over = True
            if self.moves_left <= 0 and not self._check_win_condition():
                self.score -= 50
                reward -= 50
                self.win_message = "OUT OF MOVES"
            else: # Win condition must have been met
                self.score += 100
                reward += 100
                self.win_message = "PERFECT!"

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _setup_zones_and_balls(self):
        self.grid.fill(-1)
        self.balls.clear()
        
        zone_width = 4
        zone_height = self.GRID_ROWS
        spacing = (self.GRID_COLS - zone_width * len(self.COLORS)) // (len(self.COLORS) + 1)
        
        self.target_zones = {}
        self.zone_centers = {}
        for i, color_name in enumerate(self.COLORS.keys()):
            x_start = spacing + i * (zone_width + spacing)
            self.target_zones[color_name] = pygame.Rect(
                x_start, 0, zone_width, zone_height
            )
            self.zone_centers[color_name] = self._grid_to_pixel(
                (x_start + zone_width / 2, zone_height / 2)
            )

        possible_positions = [
            (x, y) for x in range(self.GRID_COLS) for y in range(self.GRID_ROWS)
        ]
        
        # Ensure balls don't start in a completed state
        valid_positions = []
        for pos in possible_positions:
            in_correct_zone = False
            for color_name, zone in self.target_zones.items():
                if zone.collidepoint(pos):
                    # For simplicity, we just avoid all zones for initial placement
                    in_correct_zone = True
                    break
            if not in_correct_zone:
                valid_positions.append(pos)
        
        self.np_random.shuffle(valid_positions)
        
        ball_id = 0
        for color_name in self.COLORS.keys():
            for _ in range(self.NUM_BALLS_PER_COLOR):
                if not valid_positions:
                    raise RuntimeError("Not enough valid positions to place balls.")
                
                grid_pos = valid_positions.pop(0)
                self.balls.append({
                    "id": ball_id,
                    "color_name": color_name,
                    "color": self.COLORS[color_name],
                    "grid_pos": grid_pos,
                })
                self.grid[grid_pos] = ball_id
                ball_id += 1
    
    def _handle_input(self, movement, space_held):
        # --- Cursor Movement ---
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1  # Down
        elif movement == 3: self.cursor_pos[0] -= 1  # Left
        elif movement == 4: self.cursor_pos[0] += 1  # Right
        
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_COLS - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_ROWS - 1)

        # --- Pick up Ball ---
        is_space_press = space_held and not self.last_space_held
        if is_space_press and self.held_ball_id is None:
            ball_id_at_cursor = self.grid[tuple(self.cursor_pos)]
            if ball_id_at_cursor != -1:
                self.held_ball_id = ball_id_at_cursor
                # sfx: pickup_sound

    def _process_drop(self, space_held):
        is_space_press = space_held and not self.last_space_held
        move_made = False
        reward = 0

        if is_space_press and self.held_ball_id is not None:
            held_ball = self.balls[self.held_ball_id]
            origin_pos = held_ball["grid_pos"]
            target_pos = tuple(self.cursor_pos)

            # Check if dropping on its own spot (cancel action)
            if origin_pos == target_pos:
                self.held_ball_id = None
                self.last_space_held = space_held
                return 0, False

            # This is a valid move
            self.moves_left -= 1
            move_made = True
            
            ball_at_target_id = self.grid[target_pos]

            # Calculate reward based on distance change
            dist_before = np.linalg.norm(self._grid_to_pixel(origin_pos) - self.zone_centers[held_ball["color_name"]])
            dist_after = np.linalg.norm(self._grid_to_pixel(target_pos) - self.zone_centers[held_ball["color_name"]])
            reward += (dist_before - dist_after) * 0.01  # Small continuous reward

            # --- Perform move or swap ---
            if ball_at_target_id == -1: # Move to empty space
                self.grid[origin_pos] = -1
                self.grid[target_pos] = self.held_ball_id
                held_ball["grid_pos"] = target_pos
            else: # Swap with another ball
                other_ball = self.balls[ball_at_target_id]
                
                # Update grid
                self.grid[origin_pos] = ball_at_target_id
                self.grid[target_pos] = self.held_ball_id
                
                # Update ball positions
                held_ball["grid_pos"] = target_pos
                other_ball["grid_pos"] = origin_pos
                
                # Add reward for the other ball's movement
                dist_before_other = np.linalg.norm(self._grid_to_pixel(target_pos) - self.zone_centers[other_ball["color_name"]])
                dist_after_other = np.linalg.norm(self._grid_to_pixel(origin_pos) - self.zone_centers[other_ball["color_name"]])
                reward += (dist_before_other - dist_after_other) * 0.01
            
            # Check for placement reward
            if self.target_zones[held_ball["color_name"]].collidepoint(held_ball["grid_pos"]):
                reward += 5
            
            # sfx: drop_sound
            self._create_particles(self._grid_to_pixel(target_pos))
            self.held_ball_id = None
        
        self.last_space_held = space_held
        return reward, move_made

    def _check_win_condition(self):
        for color_name, zone in self.target_zones.items():
            for ball in self.balls:
                if ball["color_name"] == color_name:
                    if not zone.collidepoint(ball["grid_pos"]):
                        return False
        return True

    def _check_termination(self):
        if self._check_win_condition():
            return True
        if self.moves_left <= 0:
            return True
        if self.steps >= 1000:
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render target zones
        completed_zones = self._get_completed_zones()
        for color_name, zone_rect in self.target_zones.items():
            rect = pygame.Rect(
                zone_rect.x * self.CELL_SIZE,
                zone_rect.y * self.CELL_SIZE,
                zone_rect.width * self.CELL_SIZE,
                zone_rect.height * self.CELL_SIZE
            )
            pygame.draw.rect(self.screen, self.TARGET_COLORS[color_name], rect)
            
            # Render checkmark for completed zones
            if completed_zones[color_name]:
                self._draw_checkmark(rect.center)

        # Render balls
        for ball in self.balls:
            if ball["id"] != self.held_ball_id:
                pos = self._grid_to_pixel(ball["grid_pos"])
                self._draw_ball(pos, ball["color"])

        # Render held ball
        if self.held_ball_id is not None:
            held_ball = self.balls[self.held_ball_id]
            pos = self._grid_to_pixel(self.cursor_pos)
            self._draw_ball(pos, held_ball["color"], is_held=True)
            
            # Draw line from original spot
            origin_pos = self._grid_to_pixel(held_ball["grid_pos"])
            pygame.draw.line(self.screen, (100, 100, 100), origin_pos, pos, 1)

        # Render cursor
        cursor_pixel_pos = self._grid_to_pixel(self.cursor_pos)
        r = self.BALL_RADIUS + 4
        pygame.gfxdraw.aacircle(self.screen, cursor_pixel_pos[0], cursor_pixel_pos[1], r, self.COLOR_CURSOR)

        # Render particles
        self._update_and_draw_particles()

    def _render_ui(self):
        # Render moves left
        moves_text = self.font_main.render(f"Moves: {self.moves_left}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (10, 10))
        
        # Render score
        score_text = self.font_main.render(f"Score: {self.score:.0f}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(score_text, score_rect)

        # Render game over message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            end_text = self.font_big.render(self.win_message, True, self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }

    def _grid_to_pixel(self, grid_pos):
        x = int((grid_pos[0] + 0.5) * self.CELL_SIZE)
        y = int((grid_pos[1] + 0.5) * self.CELL_SIZE)
        return (x, y)

    def _draw_ball(self, pos, color, is_held=False):
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.BALL_RADIUS, color)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.BALL_RADIUS, tuple(min(255, c + 50) for c in color))
        if is_held:
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.BALL_RADIUS + 2, (255, 255, 255))

    def _get_completed_zones(self):
        completed = {color: True for color in self.COLORS}
        for ball in self.balls:
            in_correct_zone = self.target_zones[ball["color_name"]].collidepoint(ball["grid_pos"])
            if not in_correct_zone:
                completed[ball["color_name"]] = False
        return completed

    def _draw_checkmark(self, center):
        points = [
            (center[0] - 15, center[1]),
            (center[0] - 5, center[1] + 10),
            (center[0] + 15, center[1] - 10)
        ]
        pygame.draw.lines(self.screen, (150, 255, 150), False, points, 5)

    def _create_particles(self, pos):
        for _ in range(12):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(2, 5)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                'pos': list(pos),
                'vel': vel,
                'life': 1, # Since auto_advance is False, they only live one frame
                'color': (200, 200, 200)
            })

    def _update_and_draw_particles(self):
        if not self.particles:
            return
        
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], (int(p['pos'][0]), int(p['pos'][1])), 2)
        
        # Clear particles after drawing them for one frame
        self.particles.clear()
        
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
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
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Color Sorter")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    terminated = False
    
    # Action state
    movement = 0
    space_held = 0
    shift_held = 0 # Unused in this implementation
    
    print("\n" + "="*30)
    print("      MANUAL PLAY MODE")
    print("="*30)
    print(GameEnv.user_guide)
    print("Press Q or close window to quit.")
    print("="*30 + "\n")

    while not terminated:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                terminated = True

        # --- Poll keyboard for actions ---
        keys = pygame.key.get_pressed()
        
        # Movement (only one direction at a time)
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        else: movement = 0
            
        # Buttons
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        # --- Step the environment ---
        action = np.array([movement, space_held, shift_held])
        obs, reward, term, trunc, info = env.step(action)
        terminated = term
        
        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']:.0f}")

        # --- Render the observation to the display window ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit frame rate for human play

    print("\nGame Over!")
    print(f"Final Score: {info['score']:.0f}")
    
    # Keep window open for a few seconds to show final state
    end_time = pygame.time.get_ticks() + 3000
    while pygame.time.get_ticks() < end_time:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                break
        clock.tick(30)

    env.close()