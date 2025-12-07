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
        "Controls: Arrow keys to move your white square. Push colored boxes onto matching targets."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Sokoban Squares: A minimalist puzzle game. Push all boxes onto their matching colored targets before you run out of moves. Plan your pushes carefully!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_COLS, self.GRID_ROWS = 12, 8
        self.CELL_SIZE = 50
        self.GRID_WIDTH = self.GRID_COLS * self.CELL_SIZE
        self.GRID_HEIGHT = self.GRID_ROWS * self.CELL_SIZE
        self.GRID_X_OFFSET = (self.WIDTH - self.GRID_WIDTH) // 2
        self.GRID_Y_OFFSET = (self.HEIGHT - self.GRID_HEIGHT) // 2
        self.NUM_BOXES = 5
        self.MAX_MOVES = 40
        self.PUZZLE_SHUFFLE_STEPS = 60

        # Colors
        self.COLOR_BG = (25, 25, 35)
        self.COLOR_GRID = (50, 50, 60)
        self.COLOR_PLAYER = (255, 255, 255)
        self.BOX_COLORS = [
            (255, 80, 80),   # Red
            (80, 255, 80),   # Green
            (80, 120, 255),  # Blue
            (255, 255, 80),  # Yellow
            (200, 80, 255),  # Purple
        ]

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_msg = pygame.font.SysFont("monospace", 48, bold=True)

        # Game state variables (initialized in reset)
        self.player_pos = None
        self.boxes = None
        self.targets = None
        self.moves_remaining = None
        self.score = None
        self.game_over = None
        self.win_message = ""
        self.particles = []
        
        # Initialize state
        # A seed is required for the first reset call
        self.reset(seed=0)


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.moves_remaining = self.MAX_MOVES
        self.score = 0
        self.game_over = False
        self.win_message = ""
        self.particles = []

        self._generate_puzzle()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            obs = self._get_observation()
            return obs, 0, True, False, self._get_info()

        movement = action[0]
        reward = 0
        
        # Any non-noop action costs a move
        if movement != 0:
            self.moves_remaining -= 1
            reward -= 0.1 # Cost for taking a step

            dx, dy = 0, 0
            if movement == 1: dy = -1  # Up
            elif movement == 2: dy = 1   # Down
            elif movement == 3: dx = -1  # Left
            elif movement == 4: dx = 1   # Right

            player_next_x = self.player_pos[0] + dx
            player_next_y = self.player_pos[1] + dy

            # Check if next position is a box
            box_to_push = self._get_entity_at(self.boxes, player_next_x, player_next_y)

            if box_to_push:
                box_next_x = player_next_x + dx
                box_next_y = player_next_y + dy

                # Check if box can be pushed
                can_push = self._is_pos_valid(box_next_x, box_next_y)

                if can_push:
                    # Check box state before moving
                    was_on_target = box_to_push['on_target']

                    # Move box and player
                    box_to_push['pos'] = [box_next_x, box_next_y]
                    self.player_pos = [player_next_x, player_next_y]

                    # Check box state after moving
                    is_on_target = self._check_if_on_target(box_to_push)
                    box_to_push['on_target'] = is_on_target

                    # Reward logic
                    if is_on_target and not was_on_target:
                        reward += 10
                        self._create_particles(box_to_push['pos'], box_to_push['color'])
                    elif not is_on_target and was_on_target:
                        reward -= 10

            # If next position is empty, just move the player
            elif self._is_pos_valid(player_next_x, player_next_y):
                 self.player_pos = [player_next_x, player_next_y]

        self.score += reward
        terminated = self._check_termination()
        
        if terminated and not self.game_over:
            self.game_over = True
            if all(b['on_target'] for b in self.boxes):
                self.score += 100
                reward += 100
                self.win_message = "PUZZLE SOLVED!"
            else: # Ran out of moves
                self.score -= 50
                reward -= 50
                self.win_message = "OUT OF MOVES"

        obs = self._get_observation()
        return obs, reward, terminated, False, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._draw_grid()
        self._draw_targets()
        self._update_and_draw_particles()
        self._draw_boxes()
        self._draw_player()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "moves_remaining": self.moves_remaining}

    def _generate_puzzle(self):
        all_positions = [(x, y) for x in range(self.GRID_COLS) for y in range(self.GRID_ROWS)]
        self.np_random.shuffle(all_positions)
        target_positions = [list(pos) for pos in all_positions[:self.NUM_BOXES]]
        
        self.targets = [{'pos': pos, 'color': color} for pos, color in zip(target_positions, self.BOX_COLORS)]
        
        self.boxes = [{'pos': list(t['pos']), 'color': t['color'], 'on_target': True} for t in self.targets]

        for _ in range(self.PUZZLE_SHUFFLE_STEPS):
            box_idx = self.np_random.integers(0, len(self.boxes))
            box_to_move = self.boxes[box_idx]
            
            directions = [(-1,0), (1,0), (0,-1), (0,1)]
            self.np_random.shuffle(directions)
            
            for dx, dy in directions:
                pull_from_pos = [box_to_move['pos'][0] - dx, box_to_move['pos'][1] - dy]
                player_sim_pos = [box_to_move['pos'][0] - 2*dx, box_to_move['pos'][1] - 2*dy]

                # Create a temporary list of other boxes for collision checking
                other_boxes = self.boxes[:box_idx] + self.boxes[box_idx+1:]

                is_pull_valid = self._is_in_bounds(pull_from_pos[0], pull_from_pos[1]) and \
                                self._get_entity_at(other_boxes, pull_from_pos[0], pull_from_pos[1]) is None
                is_player_pos_valid = self._is_in_bounds(player_sim_pos[0], player_sim_pos[1]) and \
                                      self._get_entity_at(other_boxes, player_sim_pos[0], player_sim_pos[1]) is None and \
                                      not (player_sim_pos[0] == pull_from_pos[0] and player_sim_pos[1] == pull_from_pos[1])

                if is_pull_valid and is_player_pos_valid:
                    box_to_move['pos'] = pull_from_pos
                    break

        occupied_positions = {tuple(b['pos']) for b in self.boxes}
        empty_positions = [pos for pos in all_positions if tuple(pos) not in occupied_positions]
        if not empty_positions:
             self.reset(seed=self.np_random.integers(0, 100000))
             return
        
        player_start_idx = self.np_random.integers(0, len(empty_positions))
        self.player_pos = list(empty_positions[player_start_idx])

        for box in self.boxes:
            box['on_target'] = self._check_if_on_target(box)

    def _check_termination(self):
        if all(b['on_target'] for b in self.boxes):
            return True
        
        if self.moves_remaining <= 0:
            return True
            
        return False

    def _get_entity_at(self, entity_list, x, y):
        for entity in entity_list:
            if entity['pos'][0] == x and entity['pos'][1] == y:
                return entity
        return None

    def _is_in_bounds(self, x, y):
        return 0 <= x < self.GRID_COLS and 0 <= y < self.GRID_ROWS

    def _is_pos_valid(self, x, y):
        """Checks if a position is in bounds and not occupied by a box."""
        return self._is_in_bounds(x, y) and self._get_entity_at(self.boxes, x, y) is None

    def _check_if_on_target(self, box):
        target = self._get_entity_at(self.targets, box['pos'][0], box['pos'][1])
        return target is not None and target['color'] == box['color']

    def _grid_to_pixel(self, x, y):
        px = self.GRID_X_OFFSET + x * self.CELL_SIZE + self.CELL_SIZE // 2
        py = self.GRID_Y_OFFSET + y * self.CELL_SIZE + self.CELL_SIZE // 2
        return int(px), int(py)

    def _draw_grid(self):
        for x in range(self.GRID_COLS + 1):
            start_pos = (self.GRID_X_OFFSET + x * self.CELL_SIZE, self.GRID_Y_OFFSET)
            end_pos = (self.GRID_X_OFFSET + x * self.CELL_SIZE, self.GRID_Y_OFFSET + self.GRID_HEIGHT)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)
        for y in range(self.GRID_ROWS + 1):
            start_pos = (self.GRID_X_OFFSET, self.GRID_Y_OFFSET + y * self.CELL_SIZE)
            end_pos = (self.GRID_X_OFFSET + self.GRID_WIDTH, self.GRID_Y_OFFSET + y * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)

    def _draw_targets(self):
        for target in self.targets:
            px, py = self._grid_to_pixel(target['pos'][0], target['pos'][1])
            r, g, b = target['color']
            desaturated_color = (int(r*0.3+40), int(g*0.3+40), int(b*0.3+40))
            radius = int(self.CELL_SIZE * 0.4)
            pygame.gfxdraw.filled_circle(self.screen, px, py, radius, desaturated_color)
            pygame.gfxdraw.aacircle(self.screen, px, py, radius, target['color'])

    def _draw_boxes(self):
        size = int(self.CELL_SIZE * 0.8)
        inner_size = int(size * 0.7)
        for box in self.boxes:
            px, py = self._grid_to_pixel(box['pos'][0], box['pos'][1])
            rect = pygame.Rect(px - size//2, py - size//2, size, size)
            inner_rect = pygame.Rect(px - inner_size//2, py - inner_size//2, inner_size, inner_size)
            pygame.draw.rect(self.screen, (0,0,0), rect, border_radius=4)
            pygame.draw.rect(self.screen, box['color'], inner_rect, border_radius=3)
            if box['on_target']:
                 pygame.draw.rect(self.screen, (255,255,255), rect, width=2, border_radius=4)

    def _draw_player(self):
        size = int(self.CELL_SIZE * 0.7)
        px, py = self._grid_to_pixel(self.player_pos[0], self.player_pos[1])
        points = [
            (px, py - size//2),
            (px + size//2, py + size//2),
            (px - size//2, py + size//2)
        ]
        pygame.draw.polygon(self.screen, self.COLOR_PLAYER, points)
        pygame.gfxdraw.aapolygon(self.screen, points, (0,0,0))

    def _create_particles(self, grid_pos, color):
        px, py = self._grid_to_pixel(grid_pos[0], grid_pos[1])
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            radius = self.np_random.uniform(2, 5)
            life = self.np_random.integers(10, 20)
            self.particles.append({'pos': [px, py], 'vel': vel, 'radius': radius, 'color': color, 'life': life})

    def _update_and_draw_particles(self):
        active_particles = []
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] > 0:
                alpha = int(255 * (p['life'] / 20))
                color = (*p['color'], alpha)
                temp_surf = pygame.Surface((int(p['radius'])*2, int(p['radius'])*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (int(p['radius']), int(p['radius'])), int(p['radius']))
                self.screen.blit(temp_surf, (int(p['pos'][0] - p['radius']), int(p['pos'][1] - p['radius'])))
                active_particles.append(p)
        self.particles = active_particles
        
    def _render_ui(self):
        # Moves remaining
        moves_text = f"Moves: {self.moves_remaining}"
        text_surf = self.font_ui.render(moves_text, True, (255, 255, 255))
        bg_rect = pygame.Rect(5, 5, text_surf.get_width() + 10, text_surf.get_height() + 4)
        pygame.draw.rect(self.screen, (0,0,0), bg_rect, border_radius=5)
        self.screen.blit(text_surf, (10, 7))

        # Game over message
        if self.game_over:
            msg_surf = self.font_msg.render(self.win_message, True, (255, 255, 80))
            msg_rect = msg_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            self.screen.blit(overlay, (0, 0))

            self.screen.blit(msg_surf, msg_rect)

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game directly
    # It will not run in a headless environment
    try:
        os.environ.pop("SDL_VIDEODRIVER")
    except KeyError:
        pass
        
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    done = False
    
    # Use a dummy screen for display if running as a script
    display_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Sokoban Squares")
    
    clock = pygame.time.Clock()
    
    action = env.action_space.sample()
    action[0] = 0 # Start with no-op

    while not done:
        movement_action = 0 # No-op by default
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
                break
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    movement_action = 1
                elif event.key == pygame.K_DOWN:
                    movement_action = 2
                elif event.key == pygame.K_LEFT:
                    movement_action = 3
                elif event.key == pygame.K_RIGHT:
                    movement_action = 4
                elif event.key == pygame.K_r: # Reset
                    obs, info = env.reset()
                    movement_action = 0
        if done:
            break

        if movement_action != 0:
            action[0] = movement_action
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Action: {action[0]}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Moves Left: {info['moves_remaining']}, Terminated: {terminated}")
            if terminated:
                # Update display to show final state
                frame = np.transpose(obs, (1, 0, 2))
                surf = pygame.surfarray.make_surface(frame)
                display_screen.blit(surf, (0, 0))
                pygame.display.flip()
                
                pygame.time.wait(2000)
                obs, info = env.reset()

        # Update display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(30) # Limit frame rate

    env.close()