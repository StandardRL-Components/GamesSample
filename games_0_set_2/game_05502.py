
# Generated: 2025-08-28T05:12:51.720690
# Source Brief: brief_05502.md
# Brief Index: 5502

        
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
        "Controls: Use arrow keys to push all boxes simultaneously. Goal: Match boxes to their colored zones."
    )

    game_description = (
        "A minimalist puzzle game. Push all colored boxes to their matching zones within 25 moves."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and grid dimensions
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.GRID_WIDTH = 14
        self.GRID_HEIGHT = 8
        self.CELL_SIZE = 40
        self.GRID_X_OFFSET = (self.SCREEN_WIDTH - self.GRID_WIDTH * self.CELL_SIZE) // 2
        self.GRID_Y_OFFSET = (self.SCREEN_HEIGHT - self.GRID_HEIGHT * self.CELL_SIZE) // 2

        # Colors
        self.COLOR_BG = (25, 28, 32)
        self.COLOR_GRID = (45, 50, 56)
        self.BOX_COLORS = [
            (255, 95, 95), (255, 166, 95), (255, 235, 95), (166, 255, 95),
            (95, 255, 166), (95, 166, 255), (166, 95, 255), (255, 95, 166)
        ]

        # Game constants
        self.NUM_BOXES = 8
        self.MAX_MOVES = 25
        self.MAX_STEPS = 1000

        # Reward structure
        self.REWARD_MOVE = -0.1
        self.REWARD_BLOCKED_PUSH = -0.2
        self.REWARD_BOX_IN_ZONE = 1.0
        self.REWARD_WIN = 100.0
        self.REWARD_LOSE = -10.0

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
        self.font_ui = pygame.font.Font(None, 32)
        self.font_win_lose = pygame.font.Font(None, 64)

        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.moves_left = 0
        self.boxes = []
        self.zones = []
        self.num_correct_boxes_last_step = 0
        self.particles = []

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset game state
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.moves_left = self.MAX_MOVES
        self.particles = []

        # Generate a new puzzle
        all_cells = [(x, y) for x in range(self.GRID_WIDTH) for y in range(self.GRID_HEIGHT)]
        self.np_random.shuffle(all_cells)

        # Place zones and boxes
        self.zones = []
        self.boxes = []
        used_colors = list(self.BOX_COLORS)
        self.np_random.shuffle(used_colors)

        for i in range(self.NUM_BOXES):
            zone_pos = all_cells.pop()
            box_pos = all_cells.pop()
            color = used_colors[i]
            
            self.zones.append({"pos": zone_pos, "color": color})
            self.boxes.append({"pos": list(box_pos), "color": color})

        self.num_correct_boxes_last_step = self._count_correct_boxes()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = 0.0
        blocked_pushes = 0
        
        # Process only if a move action is given
        if movement in [1, 2, 3, 4]:
            self.moves_left -= 1
            reward += self.REWARD_MOVE

            # Determine push direction and sort boxes for correct collision order
            if movement == 1: # Up
                dx, dy = 0, -1
                self.boxes.sort(key=lambda b: b['pos'][1])
            elif movement == 2: # Down
                dx, dy = 0, 1
                self.boxes.sort(key=lambda b: b['pos'][1], reverse=True)
            elif movement == 3: # Left
                dx, dy = -1, 0
                self.boxes.sort(key=lambda b: b['pos'][0])
            else: # Right (movement == 4)
                dx, dy = 1, 0
                self.boxes.sort(key=lambda b: b['pos'][0], reverse=True)

            # --- Two-pass movement logic ---
            # Pass 1: Determine new positions without modifying state
            box_positions = {tuple(b['pos']) for b in self.boxes}
            new_positions = {} # Store intended moves: id -> new_pos
            
            for i, box in enumerate(self.boxes):
                current_pos = tuple(box['pos'])
                target_pos = (current_pos[0] + dx, current_pos[1] + dy)
                
                # Check for wall collision
                is_wall_collision = not (0 <= target_pos[0] < self.GRID_WIDTH and 0 <= target_pos[1] < self.GRID_HEIGHT)
                # Check for box collision
                is_box_collision = target_pos in box_positions
                
                if is_wall_collision or is_box_collision:
                    new_positions[i] = current_pos # Doesn't move
                    blocked_pushes += 1
                else:
                    new_positions[i] = target_pos # Will move
                    # Update occupied set for subsequent boxes in this step
                    box_positions.remove(current_pos)
                    box_positions.add(target_pos)

            # Pass 2: Apply new positions and create particles
            for i, box in enumerate(self.boxes):
                old_pos = tuple(box['pos'])
                new_pos = new_positions[i]
                if old_pos != new_pos:
                    box['pos'] = list(new_pos)
                    # Add particles for visual feedback
                    self._create_push_particles(old_pos, (dx, dy), box['color'])

        reward += blocked_pushes * self.REWARD_BLOCKED_PUSH

        # Calculate reward for newly correct boxes
        num_correct_now = self._count_correct_boxes()
        newly_correct = num_correct_now - self.num_correct_boxes_last_step
        reward += newly_correct * self.REWARD_BOX_IN_ZONE
        self.num_correct_boxes_last_step = num_correct_now
        
        self.score += reward
        self.steps += 1
        
        # Check for termination
        terminated = False
        is_win = num_correct_now == self.NUM_BOXES
        is_loss = self.moves_left <= 0
        
        if is_win:
            reward += self.REWARD_WIN
            self.score += self.REWARD_WIN
            terminated = True
            self.game_over = True
        elif is_loss:
            reward += self.REWARD_LOSE
            self.score += self.REWARD_LOSE
            terminated = True
            self.game_over = True

        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _count_correct_boxes(self):
        correct_count = 0
        box_positions = {tuple(b['pos']): b['color'] for b in self.boxes}
        for zone in self.zones:
            pos = tuple(zone['pos'])
            if pos in box_positions and box_positions[pos] == zone['color']:
                correct_count += 1
        return correct_count

    def _create_push_particles(self, grid_pos, direction, color):
        # Create particles at the trailing edge of the moving box
        px = self.GRID_X_OFFSET + grid_pos[0] * self.CELL_SIZE + self.CELL_SIZE / 2
        py = self.GRID_Y_OFFSET + grid_pos[1] * self.CELL_SIZE + self.CELL_SIZE / 2
        
        for _ in range(5):
            vel_x = -direction[0] * self.np_random.uniform(1, 3) + self.np_random.uniform(-0.5, 0.5)
            vel_y = -direction[1] * self.np_random.uniform(1, 3) + self.np_random.uniform(-0.5, 0.5)
            life = self.np_random.integers(10, 20)
            self.particles.append({'pos': [px, py], 'vel': [vel_x, vel_y], 'life': life, 'color': color})

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        # Render grid lines
        for x in range(self.GRID_WIDTH + 1):
            start_pos = (self.GRID_X_OFFSET + x * self.CELL_SIZE, self.GRID_Y_OFFSET)
            end_pos = (self.GRID_X_OFFSET + x * self.CELL_SIZE, self.GRID_Y_OFFSET + self.GRID_HEIGHT * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)
        for y in range(self.GRID_HEIGHT + 1):
            start_pos = (self.GRID_X_OFFSET, self.GRID_Y_OFFSET + y * self.CELL_SIZE)
            end_pos = (self.GRID_X_OFFSET + self.GRID_WIDTH * self.CELL_SIZE, self.GRID_Y_OFFSET + y * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)

        # Render zones (desaturated)
        for zone in self.zones:
            rect = pygame.Rect(
                self.GRID_X_OFFSET + zone['pos'][0] * self.CELL_SIZE,
                self.GRID_Y_OFFSET + zone['pos'][1] * self.CELL_SIZE,
                self.CELL_SIZE, self.CELL_SIZE
            )
            desaturated_color = tuple(int(c * 0.4) for c in zone['color'])
            pygame.draw.rect(self.screen, desaturated_color, rect)
            pygame.gfxdraw.filled_circle(self.screen, rect.centerx, rect.centery, int(self.CELL_SIZE*0.2), tuple(int(c*0.6) for c in zone['color']))

        # Render particles
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
            else:
                size = max(1, int(p['life'] / 4))
                pygame.draw.rect(self.screen, p['color'], (int(p['pos'][0]), int(p['pos'][1]), size, size))

        # Render boxes (bright, with 3D effect)
        for box in self.boxes:
            rect = pygame.Rect(
                self.GRID_X_OFFSET + box['pos'][0] * self.CELL_SIZE,
                self.GRID_Y_OFFSET + box['pos'][1] * self.CELL_SIZE,
                self.CELL_SIZE, self.CELL_SIZE
            )
            border_rect = rect.inflate(-8, -8)
            shadow_color = tuple(int(c * 0.6) for c in box['color'])
            highlight_color = tuple(min(255, int(c * 1.2)) for c in box['color'])

            pygame.draw.rect(self.screen, shadow_color, border_rect.move(2, 2))
            pygame.draw.rect(self.screen, box['color'], border_rect)

        # Render UI
        moves_text = self.font_ui.render(f"Moves: {self.moves_left}", True, (220, 220, 220))
        self.screen.blit(moves_text, (20, 20))

        correct_text = self.font_ui.render(f"Placed: {self.num_correct_boxes_last_step}/{self.NUM_BOXES}", True, (220, 220, 220))
        self.screen.blit(correct_text, (self.SCREEN_WIDTH - correct_text.get_width() - 20, 20))

        # Render win/loss message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self._count_correct_boxes() == self.NUM_BOXES:
                end_text = self.font_win_lose.render("PUZZLE SOLVED!", True, (100, 255, 100))
            else:
                end_text = self.font_win_lose.render("OUT OF MOVES", True, (255, 100, 100))

            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "correctly_placed": self.num_correct_boxes_last_step,
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()

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
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    
    # Create a window to display the game
    pygame.display.set_caption("Sokoban Gym Environment")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    action = np.array([0, 0, 0]) # Start with no-op
    
    print("\n" + "="*30)
    print("MANUAL PLAY MODE")
    print(env.user_guide)
    print("Press R to reset, Q to quit.")
    print("="*30 + "\n")

    running = True
    while running:
        # Render the current state
        frame = env._get_observation()
        frame_surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        screen.blit(frame_surface, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Game Over! Final Info: {info}")
            # Wait for reset or quit
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    if event.key == pygame.K_r:
                        obs, info = env.reset()
                        terminated = False
                        print("\n--- NEW GAME ---")
            continue

        # Get user input
        action_taken = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                move_action = 0 # no-op
                if event.key == pygame.K_UP:
                    move_action = 1
                elif event.key == pygame.K_DOWN:
                    move_action = 2
                elif event.key == pygame.K_LEFT:
                    move_action = 3
                elif event.key == pygame.K_RIGHT:
                    move_action = 4
                elif event.key == pygame.K_r:
                    obs, info = env.reset()
                    terminated = False
                    print("\n--- NEW GAME ---")
                    action_taken = True # To skip step
                    continue
                elif event.key == pygame.K_q:
                    running = False
                    continue

                if move_action != 0:
                    action = np.array([move_action, 0, 0])
                    obs, reward, terminated, truncated, info = env.step(action)
                    print(f"Step: {info['steps']}, Action: {move_action}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Correct: {info['correctly_placed']}")
                    action_taken = True
        
    env.close()