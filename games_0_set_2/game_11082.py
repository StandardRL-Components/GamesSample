import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T16:34:05.183101
# Source Brief: brief_01082.md
# Brief Index: 1082
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A 2D Puzzle Platformer Gymnasium environment.

    The agent controls a robot avatar to collect colored energy cells.
    Collecting a cell activates all robots of the same color.
    Activating all robots of a specific color can open corresponding doors.
    The goal is to rescue/activate all robots on the level within a step limit.

    **Visuals:**
    - Retro pixel art style with modern visual effects like glows and particles.
    - High-contrast colors for interactive elements.
    - Clear UI displaying game state.

    **Action Space:** MultiDiscrete([5, 2, 2])
    - `actions[0]`: Movement (0: none, 1: up, 2: down, 3: left, 4: right)
    - `actions[1]`: Unused (space button)
    - `actions[2]`: Unused (shift button)

    **Observation Space:** Box(0, 255, (400, 640, 3), uint8)
    - An RGB image of the game screen.

    **Rewards:**
    - -0.01 per step (encourages efficiency)
    - +1.0 for collecting an energy cell
    - +5.0 for each robot activated
    - +100.0 for winning the game (rescuing all robots)
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Control a robot to collect energy cells, activate other robots of the same color, "
        "and open doors to rescue them all before running out of steps."
    )
    user_guide = "Controls: Use the arrow keys (↑↓←→) to move the robot."
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    TILE_SIZE = 20
    GRID_WIDTH = SCREEN_WIDTH // TILE_SIZE
    GRID_HEIGHT = SCREEN_HEIGHT // TILE_SIZE
    MAX_STEPS = 500

    # --- Colors ---
    COLOR_BG = (20, 25, 40)
    COLOR_WALL = (45, 55, 75)
    COLOR_PLAYER = (0, 255, 255)
    COLOR_TEXT = (220, 220, 240)
    
    PALETTE = [
        # (Inactive, Active, Cell)
        ((150, 50, 50), (255, 80, 80), (255, 100, 100)), # 0: Red
        ((50, 50, 150), (80, 80, 255), (100, 100, 255)), # 1: Blue
        ((50, 150, 50), (80, 255, 80), (100, 255, 100)), # 2: Green
    ]

    LEVEL_LAYOUT = [
        "################################",
        "#P r .......................... #",
        "# ###########1##################",
        "# R .......................... #",
        "# # ########################## #",
        "# R # b .................. 2 # #",
        "# # # ###################### # #",
        "# . # B .................... # #",
        "# . # B .................... # #",
        "# . ######################## # #",
        "# . ........................ # #",
        "# . ........................ # #",
        "# . ........................ # #",
        "# . ........................ # #",
        "# . ........................ # #",
        "# . ........................ # #",
        "# . ........................ # #",
        "# . ........................ # #",
        "# . ........................ # #",
        "################################",
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.render_mode = render_mode
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 18, bold=True)
        
        self.player_pos = np.array([0, 0])
        self.walls = []
        self.robots = []
        self.cells = []
        self.doors = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.particles = []
        
        self._generate_level()

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        self.steps += 1
        reward = -0.01  # Cost of existing

        # --- Update game logic ---
        self._handle_movement(movement)
        
        reward_from_interactions = self._handle_interactions()
        reward += reward_from_interactions

        self._update_particles()

        # --- Check termination conditions ---
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS and not terminated
        
        if terminated and self.total_robots_rescued() == len(self.robots):
            reward += 100.0 # Victory bonus
        
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _generate_level(self):
        self.walls, self.robots, self.cells, self.doors = [], [], [], []
        for y, row in enumerate(self.LEVEL_LAYOUT):
            for x, char in enumerate(row):
                pos = np.array([x, y])
                if char == '#':
                    self.walls.append(pos)
                elif char == 'P':
                    self.player_pos = pos
                elif char.isupper(): # Robots
                    color_id = "RBG".find(char)
                    self.robots.append({'pos': pos, 'color_id': color_id, 'is_active': False, 'activation_progress': 0.0})
                elif char.islower(): # Cells
                    color_id = "rbg".find(char)
                    self.cells.append({'pos': pos, 'color_id': color_id})
                elif char.isdigit(): # Doors
                    color_key = int(char) - 1
                    self.doors.append({'pos': pos, 'color_key': color_key, 'is_open': False})

    def _handle_movement(self, movement):
        new_pos = self.player_pos.copy()
        if movement == 1: new_pos[1] -= 1  # Up
        elif movement == 2: new_pos[1] += 1  # Down
        elif movement == 3: new_pos[0] -= 1  # Left
        elif movement == 4: new_pos[0] += 1  # Right
        
        if self._is_walkable(new_pos):
            self.player_pos = new_pos

    def _is_walkable(self, pos):
        if not (0 <= pos[0] < self.GRID_WIDTH and 0 <= pos[1] < self.GRID_HEIGHT):
            return False
        if any(np.array_equal(pos, wall) for wall in self.walls):
            return False
        for door in self.doors:
            if np.array_equal(pos, door['pos']) and not door['is_open']:
                return False
        return True

    def _handle_interactions(self):
        reward = 0
        
        # Cell collection
        cell_to_remove = None
        for cell in self.cells:
            if np.array_equal(self.player_pos, cell['pos']):
                reward += 1.0
                # SFX: Cell pickup
                self._spawn_particles(cell['pos'], self.PALETTE[cell['color_id']][2], 20)
                
                # Activate robots of the same color
                robots_activated_this_turn = 0
                for robot in self.robots:
                    if robot['color_id'] == cell['color_id'] and not robot['is_active']:
                        robot['is_active'] = True
                        robots_activated_this_turn += 1
                        # SFX: Robot activate
                reward += robots_activated_this_turn * 5.0
                
                self._check_doors()
                cell_to_remove = cell
                break
        
        if cell_to_remove:
            self.cells.remove(cell_to_remove)
            
        return reward

    def _check_doors(self):
        for door in self.doors:
            if not door['is_open']:
                required_color = door['color_key']
                all_robots_of_color_active = all(
                    r['is_active'] for r in self.robots if r['color_id'] == required_color
                )
                if all_robots_of_color_active:
                    door['is_open'] = True
                    # SFX: Door open
                    self._spawn_particles(door['pos'], self.PALETTE[door['color_key']][1], 30, is_burst=True)

    def _check_termination(self):
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        if self.total_robots_rescued() == len(self.robots):
            self.game_over = True
            return True
        return False

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
            "robots_rescued": self.total_robots_rescued(),
            "robots_total": len(self.robots),
        }

    def total_robots_rescued(self):
        return sum(1 for r in self.robots if r['is_active'])

    # --- Rendering Methods ---

    def _render_game(self):
        # Draw static elements
        for wall_pos in self.walls:
            self._draw_tile(wall_pos, self.COLOR_WALL)
        for door in self.doors:
            self._draw_door(door)

        # Draw dynamic elements
        for cell in self.cells:
            self._draw_cell(cell)
        for robot in self.robots:
            self._draw_robot(robot)

        self._draw_particles()
        self._draw_player()

    def _draw_tile(self, pos, color, border_color=None):
        rect = pygame.Rect(pos[0] * self.TILE_SIZE, pos[1] * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
        pygame.draw.rect(self.screen, color, rect)
        if border_color:
            pygame.draw.rect(self.screen, border_color, rect, 1)

    def _draw_player(self):
        px, py = (self.player_pos[0] + 0.5) * self.TILE_SIZE, (self.player_pos[1] + 0.5) * self.TILE_SIZE
        radius = self.TILE_SIZE // 2 - 2
        
        # Glow effect
        for i in range(radius, radius + 5):
            alpha = 60 - (i - radius) * 12
            if alpha > 0:
                pygame.gfxdraw.aacircle(self.screen, int(px), int(py), i, (*self.COLOR_PLAYER, alpha))

        pygame.gfxdraw.filled_circle(self.screen, int(px), int(py), radius, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, int(px), int(py), radius, self.COLOR_PLAYER)

    def _draw_robot(self, robot):
        is_active = robot['is_active']
        if is_active and robot['activation_progress'] < 1.0:
            robot['activation_progress'] = min(1.0, robot['activation_progress'] + 0.05)

        progress = robot['activation_progress']
        inactive_color, active_color, _ = self.PALETTE[robot['color_id']]
        
        color = tuple(int(inactive_color[i] + (active_color[i] - inactive_color[i]) * progress) for i in range(3))

        center_x = (robot['pos'][0] + 0.5) * self.TILE_SIZE
        center_y = (robot['pos'][1] + 0.5) * self.TILE_SIZE
        size = self.TILE_SIZE - 6
        rect = pygame.Rect(center_x - size // 2, center_y - size // 2, size, size)
        
        # Body
        pygame.draw.rect(self.screen, color, rect, border_radius=2)
        
        # Eye
        eye_color = (255, 255, 255) if is_active else (10, 10, 10)
        eye_size = int(2 + 2 * progress)
        pygame.draw.circle(self.screen, eye_color, (int(center_x), int(center_y)), eye_size)

        # Active glow
        if is_active:
            glow_radius = int(size // 2 + 3 * math.sin(self.steps * 0.1 + robot['pos'][0]) * progress)
            alpha = int(80 * progress)
            if glow_radius > 0 and alpha > 0:
                 pygame.gfxdraw.aacircle(self.screen, int(center_x), int(center_y), glow_radius, (*active_color, alpha))


    def _draw_cell(self, cell):
        center_x = (cell['pos'][0] + 0.5) * self.TILE_SIZE
        center_y = (cell['pos'][1] + 0.5) * self.TILE_SIZE
        
        radius = self.TILE_SIZE // 4
        color = self.PALETTE[cell['color_id']][2]

        # Pulsing glow
        pulse = (math.sin(self.steps * 0.2 + cell['pos'][0] * 0.5) + 1) / 2 # 0 to 1
        glow_radius = int(radius + 3 + pulse * 3)
        alpha = int(50 + pulse * 40)
        pygame.gfxdraw.aacircle(self.screen, int(center_x), int(center_y), glow_radius, (*color, alpha))

        pygame.gfxdraw.filled_circle(self.screen, int(center_x), int(center_y), radius, color)
        pygame.gfxdraw.aacircle(self.screen, int(center_x), int(center_y), radius, (255, 255, 255))


    def _draw_door(self, door):
        if door['is_open']:
            return # Don't draw open doors, revealing the background
        
        color_id = door['color_key']
        color = self.PALETTE[color_id][0] # Inactive robot color
        rect = pygame.Rect(door['pos'][0] * self.TILE_SIZE, door['pos'][1] * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_WALL, rect)
        
        # Draw a symbol on the door
        symbol_rect = rect.inflate(-self.TILE_SIZE//2, -self.TILE_SIZE//2)
        pygame.draw.rect(self.screen, color, symbol_rect, border_radius=2)
        pygame.draw.rect(self.screen, self.PALETTE[color_id][1], symbol_rect, 1, border_radius=2)

    def _render_ui(self):
        robots_left_text = f"RESCUED: {self.total_robots_rescued()}/{len(self.robots)}"
        steps_text = f"STEPS: {self.steps}/{self.MAX_STEPS}"
        score_text = f"SCORE: {self.score:.2f}"

        text_surface_robots = self.font.render(robots_left_text, True, self.COLOR_TEXT)
        text_surface_steps = self.font.render(steps_text, True, self.COLOR_TEXT)
        text_surface_score = self.font.render(score_text, True, self.COLOR_TEXT)
        
        self.screen.blit(text_surface_robots, (10, 5))
        self.screen.blit(text_surface_steps, (self.SCREEN_WIDTH - text_surface_steps.get_width() - 10, 5))
        self.screen.blit(text_surface_score, (self.SCREEN_WIDTH - text_surface_score.get_width() - 10, 25))

    # --- Particle System ---
    def _spawn_particles(self, pos, color, count, is_burst=False):
        center_x = (pos[0] + 0.5) * self.TILE_SIZE
        center_y = (pos[1] + 0.5) * self.TILE_SIZE
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 3) if is_burst else random.uniform(0.5, 1.5)
            velocity = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifetime = random.randint(20, 40)
            self.particles.append({
                'pos': [center_x, center_y],
                'vel': velocity,
                'life': lifetime,
                'max_life': lifetime,
                'color': color
            })
    
    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _draw_particles(self):
        for p in self.particles:
            life_ratio = p['life'] / p['max_life']
            radius = int(life_ratio * 3)
            if radius > 0:
                alpha_color = (*p['color'], int(255 * life_ratio))
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), radius, alpha_color)

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It will create a window and render the game
    os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Robot Rescue")
    clock = pygame.time.Clock()

    done = False
    total_reward = 0
    
    print("\n--- Manual Control ---")
    print("Arrows: Move")
    print("R: Reset")
    print("Q: Quit")
    print("--------------------")

    while not done:
        action = [0, 0, 0] # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    done = True
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
                    print("--- Environment Reset ---")
                
                # Note: Holding keys is not supported in this simple loop
                if event.key == pygame.K_UP: action[0] = 1
                elif event.key == pygame.K_DOWN: action[0] = 2
                elif event.key == pygame.K_LEFT: action[0] = 3
                elif event.key == pygame.K_RIGHT: action[0] = 4

        # Only step if a move action was taken
        if action[0] != 0:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            print(f"Step: {info['steps']}, Action: {action}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}, Terminated: {terminated}")

            if terminated or truncated:
                print("--- Episode Finished ---")
                if terminated and info['robots_rescued'] == info['robots_total']:
                    print("Victory! All robots rescued.")
                else:
                    print("Game Over. Ran out of steps.")
                print(f"Final Score: {info['score']:.2f}")
                # In a real scenario, you'd reset here. For manual play, we wait for 'R'.
        
        # Render the observation from the environment to the display window
        # The main loop needs to render the latest observation regardless of action
        latest_obs = env._get_observation()
        draw_surface = pygame.surfarray.make_surface(np.transpose(latest_obs, (1, 0, 2)))
        screen.blit(draw_surface, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit frame rate

    pygame.quit()