import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import namedtuple
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys (↑, ↓, ←, →) to navigate the maze. "
        "Reach the green exit to advance. Avoid the red ghosts."
    )

    game_description = (
        "Navigate a procedurally generated haunted maze, avoiding ghosts, "
        "to reach the exit within a time limit. Complete 3 stages to win."
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAZE_WIDTH = 20
    MAZE_HEIGHT = 12
    CELL_SIZE = 32
    WALL_THICKNESS = 4

    PLAYER_RADIUS = CELL_SIZE // 4
    GHOST_RADIUS = CELL_SIZE // 4

    MOVE_DURATION = 5  # frames to move one tile

    MAX_STAGES = 3
    TIME_PER_STAGE = 60 * 30  # 60 seconds at 30 FPS
    MAX_EPISODE_STEPS = 2000
    NUM_GHOSTS = 3

    # Colors
    COLOR_BG = (10, 10, 20)
    COLOR_WALL = (40, 40, 60)
    COLOR_PLAYER = (255, 255, 255)
    COLOR_PLAYER_GLOW = (200, 200, 255, 60)
    COLOR_EXIT = (0, 255, 150)
    COLOR_EXIT_GLOW = (0, 255, 150, 80)
    COLOR_GHOST = (255, 50, 50)
    COLOR_GHOST_GLOW = (255, 50, 50, 100)

    # Data structure for ghosts
    Ghost = namedtuple("Ghost", ["pos", "visual_pos", "path", "path_index", "direction", "speed"])
    Vec2 = namedtuple("Vec2", ["x", "y"])

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
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 28)

        self.maze_offset = self.Vec2(
            (self.SCREEN_WIDTH - self.MAZE_WIDTH * self.CELL_SIZE) // 2,
            (self.SCREEN_HEIGHT - self.MAZE_HEIGHT * self.CELL_SIZE) // 2
        )

        # State variables are initialized in reset()
        self.maze = None
        self.player_pos = None
        self.player_visual_pos = None
        self.player_move_timer = 0
        self.player_move_start_pos = None
        self.player_move_target_pos = None
        self.exit_pos = None
        self.ghosts = []
        self.stage = 1
        self.time_left = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.np_random = None
        self.prev_dist_to_exit = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.stage = 1

        self._setup_stage()

        return self._get_observation(), self._get_info()

    def _setup_stage(self):
        """Initializes state for a new maze/stage."""
        self.time_left = self.TIME_PER_STAGE
        self.maze = self._generate_maze(self.MAZE_WIDTH, self.MAZE_HEIGHT)

        # Player setup
        self.player_pos = self.Vec2(0, 0)
        self.player_visual_pos = self._grid_to_pixel(self.player_pos)
        self.player_move_timer = 0

        # Exit setup
        self.exit_pos = self.Vec2(self.MAZE_WIDTH - 1, self.MAZE_HEIGHT - 1)

        # Ghost setup
        self.ghosts = []
        used_areas = []
        base_speed = 0.5 + 0.2 * (self.stage - 1)

        for _ in range(self.NUM_GHOSTS):
            patrol_rect = self._find_ghost_patrol_area(used_areas)
            if patrol_rect:
                used_areas.append(patrol_rect)
                px, py, w, h = patrol_rect

                # Create patrol path from rectangle
                path = []
                if w > 1 or h > 1:
                    for i in range(w): path.append(self.Vec2(px + i, py))
                    for i in range(1, h): path.append(self.Vec2(px + w - 1, py + i))
                    for i in range(1, w): path.append(self.Vec2(px + w - 1 - i, py + h - 1))
                    for i in range(1, h - 1): path.append(self.Vec2(px, py + h - 1 - i))
                else: # 1x1 patrol area
                    path.append(self.Vec2(px, py))

                if not path: # Should not happen with the logic above, but as a safeguard
                    start_pos = self.Vec2(px, py)
                    path = [start_pos]
                else:
                    start_pos = path[0]

                visual_pos = self._grid_to_pixel(start_pos)

                self.ghosts.append(self.Ghost(
                    pos=start_pos,
                    visual_pos=visual_pos,
                    path=path,
                    path_index=0,
                    direction=1,
                    speed=base_speed
                ))

        self.prev_dist_to_exit = self._manhattan_distance(self.player_pos, self.exit_pos)

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(30)

        movement = action[0]

        reward = -0.01  # Small penalty for each step to encourage speed
        terminated = False
        truncated = False

        if not self.game_over:
            self._handle_input(movement)
            self._update_game_state()

            # --- Reward Calculation ---
            new_dist_to_exit = self._manhattan_distance(self.player_pos, self.exit_pos)
            if new_dist_to_exit < self.prev_dist_to_exit:
                reward += 0.1
            elif new_dist_to_exit > self.prev_dist_to_exit:
                reward -= 0.1 # Small penalty for moving away
            self.prev_dist_to_exit = new_dist_to_exit

            # --- Event Checking ---
            if self._check_ghost_collision():
                reward = -10
                terminated = True
                self.game_over = True
            elif self.time_left <= 0:
                reward = -10
                terminated = True
                self.game_over = True
            elif self.player_pos == self.exit_pos and self.player_move_timer == 0:
                reward += 10
                self.score += 10
                self.stage += 1
                if self.stage > self.MAX_STAGES:
                    reward += 100
                    terminated = True
                    self.game_over = True
                else:
                    self._setup_stage()  # Next level

        self.steps += 1
        if self.steps >= self.MAX_EPISODE_STEPS:
            truncated = True

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, movement):
        if self.player_move_timer > 0:
            return  # Player is already moving

        target_pos = self.Vec2(self.player_pos.x, self.player_pos.y)
        if movement == 1:  # Up
            target_pos = self.Vec2(self.player_pos.x, self.player_pos.y - 1)
        elif movement == 2:  # Down
            target_pos = self.Vec2(self.player_pos.x, self.player_pos.y + 1)
        elif movement == 3:  # Left
            target_pos = self.Vec2(self.player_pos.x - 1, self.player_pos.y)
        elif movement == 4:  # Right
            target_pos = self.Vec2(self.player_pos.x + 1, self.player_pos.y)

        if target_pos != self.player_pos and self._is_path(self.player_pos, target_pos):
            self.player_move_start_pos = self._grid_to_pixel(self.player_pos)
            self.player_move_target_pos = self._grid_to_pixel(target_pos)
            self.player_pos = target_pos
            self.player_move_timer = self.MOVE_DURATION

    def _update_game_state(self):
        # Update player movement interpolation
        if self.player_move_timer > 0:
            self.player_move_timer -= 1
            t = 1.0 - (self.player_move_timer / self.MOVE_DURATION)
            self.player_visual_pos = self.Vec2(
                int(self.player_move_start_pos.x + (self.player_move_target_pos.x - self.player_move_start_pos.x) * t),
                int(self.player_move_start_pos.y + (self.player_move_target_pos.y - self.player_move_start_pos.y) * t)
            )
            if self.player_move_timer == 0:
                self.player_visual_pos = self.player_move_target_pos

        # Update ghosts
        new_ghosts = []
        for ghost in self.ghosts:
            if not ghost.path:
                new_ghosts.append(ghost)
                continue

            # Move ghost along its path
            current_target_node = ghost.path[ghost.path_index]
            current_target_px = self._grid_to_pixel(current_target_node)

            dx = current_target_px.x - ghost.visual_pos.x
            dy = current_target_px.y - ghost.visual_pos.y
            dist = math.hypot(dx, dy)

            move_dist = ghost.speed * (self.CELL_SIZE / self.MOVE_DURATION)

            if dist < move_dist:
                new_visual_pos = current_target_px
                # Move to next node in path
                new_path_index = (ghost.path_index + ghost.direction) % len(ghost.path)
                new_pos = ghost.path[new_path_index]
                new_ghosts.append(ghost._replace(pos=new_pos, visual_pos=new_visual_pos, path_index=new_path_index))
            else:
                new_visual_pos = self.Vec2(
                    ghost.visual_pos.x + (dx / dist) * move_dist,
                    ghost.visual_pos.y + (dy / dist) * move_dist
                )
                new_ghosts.append(ghost._replace(visual_pos=new_visual_pos))
        self.ghosts = new_ghosts

        # Update timer
        if self.time_left > 0:
            self.time_left -= 1

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
            "stage": self.stage,
            "time_left": self.time_left / 30,  # in seconds
        }

    def _render_game(self):
        # Draw maze walls
        for y in range(self.MAZE_HEIGHT):
            for x in range(self.MAZE_WIDTH):
                if not self.maze[y, x, 1]:  # South wall
                    p1 = self._grid_to_pixel(self.Vec2(x, y + 1), corner=True)
                    p2 = self._grid_to_pixel(self.Vec2(x + 1, y + 1), corner=True)
                    pygame.draw.line(self.screen, self.COLOR_WALL, p1, p2, self.WALL_THICKNESS)
                if not self.maze[y, x, 2]:  # East wall
                    p1 = self._grid_to_pixel(self.Vec2(x + 1, y), corner=True)
                    p2 = self._grid_to_pixel(self.Vec2(x + 1, y + 1), corner=True)
                    pygame.draw.line(self.screen, self.COLOR_WALL, p1, p2, self.WALL_THICKNESS)
        # Draw outer boundary
        pygame.draw.rect(self.screen, self.COLOR_WALL,
                         (self.maze_offset.x, self.maze_offset.y,
                          self.MAZE_WIDTH * self.CELL_SIZE, self.MAZE_HEIGHT * self.CELL_SIZE), self.WALL_THICKNESS)

        # Draw exit
        exit_px = self._grid_to_pixel(self.exit_pos)
        glow_size = int(self.CELL_SIZE * 0.8)
        exit_size = int(self.CELL_SIZE * 0.5)
        self._draw_glowing_rect(self.screen, self.COLOR_EXIT, self.COLOR_EXIT_GLOW, exit_px, exit_size, glow_size)

        # Draw ghosts
        for ghost in self.ghosts:
            flicker = (math.sin(self.steps * 0.5 + id(ghost)) + 1) / 2
            glow_radius = self.GHOST_RADIUS * 2 + int(flicker * 8)
            self._draw_glowing_circle(self.screen, self.COLOR_GHOST, self.COLOR_GHOST_GLOW, ghost.visual_pos,
                                      self.GHOST_RADIUS, glow_radius)

        # Draw player
        self._draw_glowing_circle(self.screen, self.COLOR_PLAYER, self.COLOR_PLAYER_GLOW, self.player_visual_pos,
                                  self.PLAYER_RADIUS, self.PLAYER_RADIUS * 3)

    def _render_ui(self):
        stage_text = self.font_large.render(f"Stage: {self.stage}/{self.MAX_STAGES}", True, self.COLOR_PLAYER)
        self.screen.blit(stage_text, (20, 10))

        time_str = f"Time: {int(self.time_left / 30):02d}"
        time_text = self.font_large.render(time_str, True,
                                           self.COLOR_EXIT if self.time_left > 10 * 30 else self.COLOR_GHOST)
        self.screen.blit(time_text, (self.SCREEN_WIDTH - time_text.get_width() - 20, 10))

    def close(self):
        pygame.quit()

    # --- Helper Functions ---

    def _grid_to_pixel(self, grid_pos, corner=False):
        if corner:
            return self.Vec2(
                self.maze_offset.x + grid_pos.x * self.CELL_SIZE,
                self.maze_offset.y + grid_pos.y * self.CELL_SIZE
            )
        return self.Vec2(
            self.maze_offset.x + grid_pos.x * self.CELL_SIZE + self.CELL_SIZE // 2,
            self.maze_offset.y + grid_pos.y * self.CELL_SIZE + self.CELL_SIZE // 2
        )

    def _manhattan_distance(self, pos1, pos2):
        return abs(pos1.x - pos2.x) + abs(pos1.y - pos2.y)

    def _generate_maze(self, width, height):
        # N, S, E, W, visited
        maze = np.zeros((height, width, 5), dtype=np.uint8)

        x, y = self.np_random.integers(0, width), self.np_random.integers(0, height)
        stack = [(x, y)]
        maze[y, x, 4] = 1

        while stack:
            x, y = stack[-1]
            neighbors = []
            # North
            if y > 0 and maze[y - 1, x, 4] == 0: neighbors.append((x, y - 1, 0, 1))
            # South
            if y < height - 1 and maze[y + 1, x, 4] == 0: neighbors.append((x, y + 1, 1, 0))
            # East
            if x < width - 1 and maze[y, x + 1, 4] == 0: neighbors.append((x + 1, y, 2, 3))
            # West
            if x > 0 and maze[y, x - 1, 4] == 0: neighbors.append((x - 1, y, 3, 2))

            if neighbors:
                nx, ny, wall_dir, neighbor_wall_dir = neighbors[self.np_random.integers(len(neighbors))]
                maze[y, x, wall_dir] = 1
                maze[ny, nx, neighbor_wall_dir] = 1
                maze[ny, nx, 4] = 1
                stack.append((nx, ny))
            else:
                stack.pop()
        return maze

    def _is_path(self, pos1, pos2):
        if not (0 <= pos2.x < self.MAZE_WIDTH and 0 <= pos2.y < self.MAZE_HEIGHT):
            return False
        if pos2.y < pos1.y: return self.maze[pos1.y, pos1.x, 0]  # North
        if pos2.y > pos1.y: return self.maze[pos1.y, pos1.x, 1]  # South
        if pos2.x > pos1.x: return self.maze[pos1.y, pos1.x, 2]  # East
        if pos2.x < pos1.x: return self.maze[pos1.y, pos1.x, 3]  # West
        return False

    def _check_ghost_collision(self):
        for ghost in self.ghosts:
            dist = math.hypot(self.player_visual_pos.x - ghost.visual_pos.x,
                              self.player_visual_pos.y - ghost.visual_pos.y)
            if dist < self.PLAYER_RADIUS + self.GHOST_RADIUS:
                return True
        return False

    def _find_ghost_patrol_area(self, used_areas):
        possible_areas = []
        # Search for 4x3 and 3x4 clearings
        for w, h in [(4, 3), (3, 4)]:
            for y in range(self.MAZE_HEIGHT - h + 1):
                for x in range(self.MAZE_WIDTH - w + 1):
                    # Check if area is clear of walls and not near player start/end
                    if (x < 2 and y < 2) or (x + w > self.MAZE_WIDTH - 2 and y + h > self.MAZE_HEIGHT - 2):
                        continue
                    
                    is_clear = True
                    for j in range(h):
                        for i in range(w):
                            # A "clear" tile for a patrol area should be a corridor or room, not a dead end.
                            if np.sum(self.maze[y + j, x + i, :4]) < 2:
                                is_clear = False
                                break
                        if not is_clear: break
                    
                    if is_clear:
                        is_overlapping = False
                        current_rect = pygame.Rect(x, y, w, h)
                        for used_rect_tuple in used_areas:
                            if current_rect.colliderect(pygame.Rect(used_rect_tuple)):
                                is_overlapping = True
                                break
                        if not is_overlapping:
                            possible_areas.append((x, y, w, h))

        if possible_areas:
            idx = self.np_random.integers(len(possible_areas))
            return possible_areas[idx]

        # Fallback for rare cases where no large areas are found
        for y in range(1, self.MAZE_HEIGHT - 1):
            for x in range(1, self.MAZE_WIDTH - 1):
                if np.sum(self.maze[y, x, :4]) > 1:
                    is_overlapping = False
                    current_rect = pygame.Rect(x, y, 1, 1)
                    for used_rect_tuple in used_areas:
                        if current_rect.colliderect(pygame.Rect(used_rect_tuple)):
                            is_overlapping = True
                            break
                    if not is_overlapping:
                        return x, y, 1, 1  # 1x1 patrol area (stand still)
        return None

    def _draw_glowing_circle(self, surface, color, glow_color, center, radius, glow_radius):
        center_int = (int(center.x), int(center.y))

        # Draw glow
        temp_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(temp_surf, glow_radius, glow_radius, glow_radius, glow_color)
        surface.blit(temp_surf, (center_int[0] - glow_radius, center_int[1] - glow_radius),
                     special_flags=pygame.BLEND_RGBA_ADD)

        # Draw main circle
        pygame.gfxdraw.aacircle(surface, center_int[0], center_int[1], radius, color)
        pygame.gfxdraw.filled_circle(surface, center_int[0], center_int[1], radius, color)

    def _draw_glowing_rect(self, surface, color, glow_color, center, size, glow_size):
        center_int = (int(center.x), int(center.y))

        # Draw glow
        glow_rect = pygame.Rect(0, 0, glow_size, glow_size)
        glow_rect.center = (glow_size // 2, glow_size // 2)
        temp_surf = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
        pygame.draw.rect(temp_surf, glow_color, glow_rect, border_radius=glow_size // 4)
        surface.blit(temp_surf, (center_int[0] - glow_size // 2, center_int[1] - glow_size // 2),
                     special_flags=pygame.BLEND_RGBA_ADD)

        # Draw main rect
        main_rect = pygame.Rect(0, 0, size, size)
        main_rect.center = center_int
        pygame.draw.rect(surface, color, main_rect, border_radius=size // 4)

    def validate_implementation(self):
        # This method is for internal verification and not part of the standard API
        print("Attempting to validate implementation...")
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
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == '__main__':
    # This block allows you to play the game directly
    # It will open a window for rendering
    os.environ["SDL_VIDEODRIVER"] = "x11"
    env = GameEnv(render_mode="rgb_array")
    
    # The validate_implementation call was in the original __init__,
    # but it's better to call it after instantiation if needed for testing.
    try:
        env.validate_implementation()
    except Exception as e:
        print(f"Validation failed: {e}")

    obs, info = env.reset()

    # Override screen to be a display surface
    env.screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Haunted Maze")

    terminated = False
    truncated = False
    total_reward = 0

    print(env.user_guide)

    # Main game loop
    while not terminated and not truncated:
        movement = 0  # No-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4

        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement, space_held, shift_held]

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation to the display
        pygame.surfarray.blit_array(env.screen, np.transpose(obs, (1, 0, 2)))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    env.close()