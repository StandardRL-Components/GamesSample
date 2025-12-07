import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A Gymnasium environment for a neon-themed, rhythm-based maze game.
    The player must navigate a procedurally generated maze to reach the exit,
    following a rhythmic pulse that indicates the optimal path.
    """
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use ↑↓←→ to move on the beat. Follow the light pulse to the green exit."
    )

    game_description = (
        "Navigate a neon maze to the rhythm. Reach the exit before time runs out, but watch out for the walls!"
    )

    auto_advance = False

    # --- Game Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 1000
    MAX_WALL_HITS = 5
    BASE_MAZE_SIZE = 5

    # --- Colors ---
    COLOR_BG = (10, 20, 40)
    COLOR_WALL = (40, 80, 180)
    COLOR_PLAYER = (255, 255, 0)
    COLOR_EXIT = (0, 255, 128)
    COLOR_PULSE = (100, 200, 255)
    COLOR_TEXT = (240, 240, 240)
    COLOR_DANGER = (255, 50, 50)
    COLOR_GAMEOVER = (200, 20, 20, 200)

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
        
        try:
            self.font_ui = pygame.font.Font(None, 24)
            self.font_gameover = pygame.font.Font(None, 72)
        except IOError:
            self.font_ui = pygame.font.SysFont("monospace", 20)
            self.font_gameover = pygame.font.SysFont("monospace", 68)

        self.level = 0
        self.maze_width = 0
        self.maze_height = 0
        self.maze = {}
        self.optimal_path = []
        self.start_pos = (0, 0)
        self.exit_pos = (0, 0)
        self.player_pos = [0, 0]
        self.steps = 0
        self.score = 0.0
        self.wall_hits = 0
        self.game_over = False
        self.wall_hit_flash = 0
        self.win_state = False
        
        # This will be initialized in reset()
        self.np_random = None

        # Seed the environment to get the np_random generator
        self.reset(seed=0)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.maze_width = self.BASE_MAZE_SIZE + self.level
        self.maze_height = self.BASE_MAZE_SIZE + self.level

        # Clamp maze size to prevent it from becoming too large for the screen
        self.maze_width = min(self.maze_width, 25)
        self.maze_height = min(self.maze_height, 15)

        self.start_pos = (0, 0)
        self.exit_pos = (self.maze_width - 1, self.maze_height - 1)
        
        self.maze = self._generate_maze(self.maze_width, self.maze_height)
        self.optimal_path = self._find_path(self.start_pos, self.exit_pos)

        self.player_pos = list(self.start_pos)
        self.steps = 0
        self.score = 0.0
        self.wall_hits = 0
        self.game_over = False
        self.win_state = False
        self.wall_hit_flash = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = 0.0
        
        old_pos = tuple(self.player_pos)
        old_dist = self._manhattan_distance(old_pos, self.exit_pos)

        # --- Process Movement ---
        dx, dy = 0, 0
        if movement == 1: dy = -1  # Up
        elif movement == 2: dy = 1   # Down
        elif movement == 3: dx = -1  # Left
        elif movement == 4: dx = 1   # Right

        if movement != 0:
            target_pos = (self.player_pos[0] + dx, self.player_pos[1] + dy)
            if self._can_move(old_pos, target_pos):
                self.player_pos[0] = target_pos[0]
                self.player_pos[1] = target_pos[1]
                # Sound: player_move.wav
            else:
                self.wall_hits += 1
                reward -= 50.0
                self.wall_hit_flash = 5 # Visual effect duration
                # Sound: wall_hit.wav

        new_pos = tuple(self.player_pos)
        new_dist = self._manhattan_distance(new_pos, self.exit_pos)

        # --- Calculate Rewards ---
        # Distance-based reward
        reward += (old_dist - new_dist)

        # Rhythmic cue reward
        if self.steps < len(self.optimal_path) - 1:
            correct_next_pos = self.optimal_path[self.steps + 1]
            if new_pos == correct_next_pos:
                reward += 5.0 # Correct move on beat
                # Sound: correct_move.wav
            elif new_pos != old_pos: # Moved, but incorrectly
                reward -= 2.0
                # Sound: incorrect_move.wav

        self.steps += 1
        
        # --- Check Termination Conditions ---
        terminated = False
        if new_pos == self.exit_pos:
            reward += 100.0
            terminated = True
            self.game_over = True
            self.win_state = True
            self.level += 1 # Increase difficulty for next game
            # Sound: win.wav
        elif self.wall_hits >= self.MAX_WALL_HITS:
            reward -= 100.0 # Extra penalty for losing this way
            terminated = True
            self.game_over = True
            self.level = max(0, self.level - 1) # Decrease difficulty
            # Sound: lose.wav
        elif self.steps >= self.MAX_STEPS:
            reward -= 100.0 # Penalty for running out of time/steps
            terminated = True
            self.game_over = True
            self.level = max(0, self.level - 1)
            # Sound: lose.wav

        self.score += reward
        
        if self.wall_hit_flash > 0:
            self.wall_hit_flash -= 1

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

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
            "wall_hits": self.wall_hits,
        }

    def _render_game(self):
        padding = 40
        maze_render_width = self.SCREEN_WIDTH - 2 * padding
        maze_render_height = self.SCREEN_HEIGHT - 2 * padding
        
        if self.maze_width == 0 or self.maze_height == 0: return

        cell_w = maze_render_width / self.maze_width
        cell_h = maze_render_height / self.maze_height
        
        offset_x = padding
        offset_y = padding

        # Render path pulse
        if not self.game_over and self.steps < len(self.optimal_path) - 1:
            pulse_pos = self.optimal_path[self.steps + 1]
            pulse_cx = int(offset_x + (pulse_pos[0] + 0.5) * cell_w)
            pulse_cy = int(offset_y + (pulse_pos[1] + 0.5) * cell_h)
            pulse_radius = int(min(cell_w, cell_h) * (0.4 + 0.1 * math.sin(pygame.time.get_ticks() * 0.01)))
            pygame.gfxdraw.filled_circle(self.screen, pulse_cx, pulse_cy, pulse_radius, self.COLOR_PULSE + (100,))
            pygame.gfxdraw.aacircle(self.screen, pulse_cx, pulse_cy, pulse_radius, self.COLOR_PULSE + (150,))

        # Render exit
        ex, ey = self.exit_pos
        exit_rect = pygame.Rect(offset_x + ex * cell_w, offset_y + ey * cell_h, cell_w, cell_h)
        glow_size = int(min(cell_w, cell_h) * (0.8 + 0.1 * math.sin(pygame.time.get_ticks() * 0.005)))
        pygame.gfxdraw.filled_circle(self.screen, int(exit_rect.centerx), int(exit_rect.centery), glow_size, self.COLOR_EXIT + (30,))
        pygame.gfxdraw.filled_circle(self.screen, int(exit_rect.centerx), int(exit_rect.centery), int(glow_size * 0.7), self.COLOR_EXIT + (50,))
        pygame.draw.rect(self.screen, self.COLOR_EXIT, exit_rect.inflate(-cell_w*0.4, -cell_h*0.4))

        # Render maze walls
        for y in range(self.maze_height):
            for x in range(self.maze_width):
                walls = self.maze.get((x, y), [True, True, True, True])
                px, py = offset_x + x * cell_w, offset_y + y * cell_h
                if walls[0]: pygame.draw.line(self.screen, self.COLOR_WALL, (px, py), (px + cell_w, py), 2) # North
                if walls[1]: pygame.draw.line(self.screen, self.COLOR_WALL, (px + cell_w, py), (px + cell_w, py + cell_h), 2) # East
                if walls[2]: pygame.draw.line(self.screen, self.COLOR_WALL, (px + cell_w, py + cell_h), (px, py + cell_h), 2) # South
                if walls[3]: pygame.draw.line(self.screen, self.COLOR_WALL, (px, py + cell_h), (px, py), 2) # West

        # Render player
        px, py = self.player_pos
        player_cx = int(offset_x + (px + 0.5) * cell_w)
        player_cy = int(offset_y + (py + 0.5) * cell_h)
        player_radius = int(min(cell_w, cell_h) * 0.3)
        
        # Glow effect
        glow_radius = int(player_radius * (1.5 + 0.2 * math.sin(pygame.time.get_ticks() * 0.02)))
        pygame.gfxdraw.filled_circle(self.screen, player_cx, player_cy, glow_radius, self.COLOR_PLAYER + (50,))
        pygame.gfxdraw.aacircle(self.screen, player_cx, player_cy, glow_radius, self.COLOR_PLAYER + (100,))
        
        pygame.gfxdraw.filled_circle(self.screen, player_cx, player_cy, player_radius, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, player_cx, player_cy, player_radius, self.COLOR_PLAYER)
        
        # Render wall hit flash
        if self.wall_hit_flash > 0:
            s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            alpha = int(150 * (self.wall_hit_flash / 5))
            s.fill(self.COLOR_DANGER + (alpha,))
            self.screen.blit(s, (0, 0))

    def _render_ui(self):
        # Wall Hits
        hits_text = self.font_ui.render(f"HITS: {self.wall_hits}/{self.MAX_WALL_HITS}", True, self.COLOR_DANGER)
        self.screen.blit(hits_text, (10, 10))

        # Steps remaining
        steps_left = self.MAX_STEPS - self.steps
        time_color = self.COLOR_TEXT if steps_left > 100 else self.COLOR_DANGER
        time_text = self.font_ui.render(f"BEATS: {steps_left}", True, time_color)
        self.screen.blit(time_text, (self.SCREEN_WIDTH - time_text.get_width() - 10, 10))

        # Score
        score_text = self.font_ui.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH // 2 - score_text.get_width() // 2, self.SCREEN_HEIGHT - 30))
        
        # Game Over Screen
        if self.game_over:
            s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            s.fill(self.COLOR_GAMEOVER)
            self.screen.blit(s, (0, 0))
            
            message = "LEVEL COMPLETE" if self.win_state else "GAME OVER"
            color = self.COLOR_EXIT if self.win_state else self.COLOR_DANGER
            
            gameover_text = self.font_gameover.render(message, True, color)
            text_rect = gameover_text.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
            self.screen.blit(gameover_text, text_rect)

    def _generate_maze(self, width, height):
        maze = {(x, y): [True, True, True, True] for x in range(width) for y in range(height)} # N, E, S, W walls
        stack = []
        visited = set()
        
        start_cell = (self.np_random.integers(0, width), self.np_random.integers(0, height))
        stack.append(start_cell)
        visited.add(start_cell)

        while stack:
            cx, cy = stack[-1]
            neighbors = []
            # Check North
            if cy > 0 and (cx, cy - 1) not in visited: neighbors.append((0, (cx, cy - 1)))
            # Check East
            if cx < width - 1 and (cx + 1, cy) not in visited: neighbors.append((1, (cx + 1, cy)))
            # Check South
            if cy < height - 1 and (cx, cy + 1) not in visited: neighbors.append((2, (cx, cy + 1)))
            # Check West
            if cx > 0 and (cx - 1, cy) not in visited: neighbors.append((3, (cx - 1, cy)))

            if neighbors:
                # FIX: self.np_random.choice cannot handle a list of nested tuples.
                # It tries to convert it to a numpy array and fails due to an inhomogeneous shape.
                # Instead, we choose a random index from the list of neighbors.
                direction, (nx, ny) = neighbors[self.np_random.integers(len(neighbors))]
                
                # Knock down walls
                if direction == 0: # North
                    maze[(cx, cy)][0] = False
                    maze[(nx, ny)][2] = False
                elif direction == 1: # East
                    maze[(cx, cy)][1] = False
                    maze[(nx, ny)][3] = False
                elif direction == 2: # South
                    maze[(cx, cy)][2] = False
                    maze[(nx, ny)][0] = False
                elif direction == 3: # West
                    maze[(cx, cy)][3] = False
                    maze[(nx, ny)][1] = False
                
                visited.add((nx, ny))
                stack.append((nx, ny))
            else:
                stack.pop()
        return maze

    def _find_path(self, start, end):
        queue = [(start, [start])]
        visited = {start}
        
        while queue:
            (x, y), path = queue.pop(0)
            if (x, y) == end:
                return path

            walls = self.maze.get((x, y), [True]*4)
            # North
            if not walls[0] and (x, y - 1) not in visited:
                visited.add((x, y - 1))
                queue.append(((x, y - 1), path + [(x, y - 1)]))
            # East
            if not walls[1] and (x + 1, y) not in visited:
                visited.add((x + 1, y))
                queue.append(((x + 1, y), path + [(x + 1, y)]))
            # South
            if not walls[2] and (x, y + 1) not in visited:
                visited.add((x, y + 1))
                queue.append(((x, y + 1), path + [(x, y + 1)]))
            # West
            if not walls[3] and (x - 1, y) not in visited:
                visited.add((x - 1, y))
                queue.append(((x - 1, y), path + [(x - 1, y)]))
        
        return [start] # Should not happen with generated maze

    def _can_move(self, current_pos, target_pos):
        cx, cy = current_pos
        tx, ty = target_pos
        
        if not (0 <= tx < self.maze_width and 0 <= ty < self.maze_height):
            return False

        walls = self.maze.get(current_pos)
        if tx > cx: return not walls[1] # Moving East
        if tx < cx: return not walls[3] # Moving West
        if ty > cy: return not walls[2] # Moving South
        if ty < cy: return not walls[0] # Moving North
        
        return False

    def _manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def close(self):
        pygame.quit()

# Example of how to run the environment
if __name__ == '__main__':
    # This block requires a display. It will not run in a truly headless environment.
    # To run the environment headlessly, you can do something like:
    # env = GameEnv()
    # obs, info = env.reset()
    # for _ in range(100):
    #   action = env.action_space.sample()
    #   obs, reward, terminated, truncated, info = env.step(action)
    #   if terminated:
    #       obs, info = env.reset()
    # env.close()

    env = GameEnv(render_mode="rgb_array")
    
    # For interactive play
    pygame.display.set_caption("Rhythm Maze")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    obs, info = env.reset()
    done = False
    
    # Game loop for interactive testing
    while not done:
        action = [0, 0, 0] # Default to no-op
        
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                done = True
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        # Since auto_advance is False, we only step on a key press
        if action[0] != 0 or any(event.type == pygame.KEYDOWN for event in events):
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated:
                print(f"Game Over! Final Score: {info['score']}")
                # Display final state for a moment before resetting
                frame = np.transpose(obs, (1, 0, 2))
                surf = pygame.surfarray.make_surface(frame)
                screen.blit(surf, (0, 0))
                pygame.display.flip()
                pygame.time.wait(2000)
                
                obs, info = env.reset()
        
        # Render the current state
        frame = np.transpose(env._get_observation(), (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(30) # Limit FPS for interactive mode

    env.close()