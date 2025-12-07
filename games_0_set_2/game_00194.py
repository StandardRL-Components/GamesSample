
# Generated: 2025-08-27T12:53:54.824275
# Source Brief: brief_00194.md
# Brief Index: 194

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = "Controls: ↑↓←→ to move your green worm. Grow longer than the red opponent before the turns run out!"

    # Must be a short, user-facing description of the game:
    game_description = "A competitive worm game. Eat food to grow, but avoid colliding with walls, yourself, or your opponent."

    # Should frames auto-advance or wait for user input?
    # Set to False for turn-based strategic gameplay.
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 20
        self.GRID_WIDTH = self.WIDTH // self.GRID_SIZE
        self.GRID_HEIGHT = self.HEIGHT // self.GRID_SIZE
        self.MAX_STEPS = 1000

        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_GRID = (40, 40, 50)
        self.COLOR_PLAYER = (0, 255, 127) # Spring Green
        self.COLOR_PLAYER_HEAD = (127, 255, 212) # Aquamarine
        self.COLOR_OPPONENT = (255, 69, 0) # OrangeRed
        self.COLOR_OPPONENT_HEAD = (255, 99, 71) # Tomato
        self.COLOR_FOOD = (255, 255, 0) # Yellow
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_PARTICLE = (255, 223, 0) # Gold

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)

        # Game state variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_body = deque()
        self.player_direction = (0, 0)
        self.opponent_body = deque()
        self.opponent_direction = (0, 0)
        self.opponent_ai_turn_direction = 1
        self.food_pos = (0, 0)
        self.particles = []
        self.last_player_dist_to_food = 0

        # Initialize state variables
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False

        # Place player
        player_start_pos = (self.GRID_WIDTH // 4, self.GRID_HEIGHT // 2)
        self.player_body = deque([player_start_pos])
        self.player_direction = (1, 0)

        # Place opponent
        opponent_start_pos = (self.GRID_WIDTH * 3 // 4, self.GRID_HEIGHT // 2)
        self.opponent_body = deque([opponent_start_pos])
        self.opponent_direction = (-1, 0)
        self.opponent_ai_turn_direction = self.np_random.choice([1, -1])

        self._spawn_food()
        self.particles = []
        self.last_player_dist_to_food = self._manhattan_distance(self.player_body[0], self.food_pos)

        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]
        # space_held = action[1] == 1
        # shift_held = action[2] == 1

        reward = 0
        terminated = False

        self._update_player(movement)
        self._update_opponent()

        player_head = self.player_body[0]
        opponent_head = self.opponent_body[0]
        
        ate_food = False
        if player_head == self.food_pos:
            reward += 10
            # Sound: eat
            self._create_particles(self.food_pos)
            self._spawn_food()
            ate_food = True
        else:
            self.player_body.pop()

        if opponent_head == self.food_pos:
            # Sound: opponent_eat
            self._create_particles(self.food_pos)
            self._spawn_food()
        else:
            self.opponent_body.pop()

        player_collided = (
            player_head in list(self.player_body)[1:] or
            player_head in self.opponent_body or
            not (0 <= player_head[0] < self.GRID_WIDTH and 0 <= player_head[1] < self.GRID_HEIGHT)
        )

        if player_collided:
            terminated = True
            reward = -50
            # Sound: death

        self.steps += 1
        if not terminated and self.steps >= self.MAX_STEPS:
            terminated = True
            if len(self.player_body) > len(self.opponent_body):
                reward += 50
            elif len(self.player_body) < len(self.opponent_body):
                reward -= 50

        if terminated:
            self.game_over = True

        if not ate_food and not terminated:
            new_dist = self._manhattan_distance(player_head, self.food_pos)
            if new_dist < self.last_player_dist_to_food:
                reward += 1
            elif new_dist > self.last_player_dist_to_food:
                reward -= 1
            self.last_player_dist_to_food = new_dist
        
        self.score = len(self.player_body) - 1

        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _update_player(self, movement):
        new_direction = self.player_direction
        if movement == 1 and self.player_direction != (0, 1): new_direction = (0, -1)
        elif movement == 2 and self.player_direction != (0, -1): new_direction = (0, 1)
        elif movement == 3 and self.player_direction != (1, 0): new_direction = (-1, 0)
        elif movement == 4 and self.player_direction != (-1, 0): new_direction = (1, 0)
        
        self.player_direction = new_direction
        
        current_head = self.player_body[0]
        new_head = (current_head[0] + self.player_direction[0], current_head[1] + self.player_direction[1])
        self.player_body.appendleft(new_head)

    def _update_opponent(self):
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        current_head = self.opponent_body[0]
        next_pos = (current_head[0] + self.opponent_direction[0], current_head[1] + self.opponent_direction[1])
        
        is_wall = not (0 <= next_pos[0] < self.GRID_WIDTH and 0 <= next_pos[1] < self.GRID_HEIGHT)
        is_self_collision = next_pos in list(self.opponent_body)

        if is_wall or is_self_collision:
            current_dir_index = directions.index(self.opponent_direction)
            new_dir_index = (current_dir_index + self.opponent_ai_turn_direction) % 4
            self.opponent_direction = directions[new_dir_index]

        new_head = (current_head[0] + self.opponent_direction[0], current_head[1] + self.opponent_direction[1])
        self.opponent_body.appendleft(new_head)

    def _spawn_food(self):
        occupied_cells = set(self.player_body) | set(self.opponent_body)
        available_cells = [
            (x, y) for x in range(self.GRID_WIDTH) for y in range(self.GRID_HEIGHT) if (x, y) not in occupied_cells
        ]
        if not available_cells:
            self.game_over = True
            self.food_pos = (-1, -1)
            return
        idx = self.np_random.integers(len(available_cells))
        self.food_pos = available_cells[idx]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid()
        self._render_food()
        self._render_worm(self.opponent_body, self.COLOR_OPPONENT, self.COLOR_OPPONENT_HEAD)
        self._render_worm(self.player_body, self.COLOR_PLAYER, self.COLOR_PLAYER_HEAD)
        self._render_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_grid(self):
        for x in range(0, self.WIDTH, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

    def _render_worm(self, body, color, head_color):
        if not body: return
        head = body[0]
        head_rect = pygame.Rect(head[0] * self.GRID_SIZE, head[1] * self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE)
        pygame.draw.rect(self.screen, head_color, head_rect)
        
        for segment in list(body)[1:]:
            seg_rect = pygame.Rect(segment[0] * self.GRID_SIZE, segment[1] * self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE)
            pygame.draw.rect(self.screen, color, seg_rect)

    def _render_food(self):
        if self.food_pos == (-1, -1): return
        pulse = (math.sin(pygame.time.get_ticks() * 0.005) + 1) / 2
        size = self.GRID_SIZE * (0.6 + pulse * 0.3)
        offset = (self.GRID_SIZE - size) / 2
        
        food_rect = pygame.Rect(
            self.food_pos[0] * self.GRID_SIZE + offset, self.food_pos[1] * self.GRID_SIZE + offset, size, size
        )
        pygame.draw.rect(self.screen, self.COLOR_FOOD, food_rect, border_radius=int(size/3))
        pygame.gfxdraw.aacircle(self.screen, int(food_rect.centerx), int(food_rect.centery), int(size*0.7), self.COLOR_FOOD)

    def _create_particles(self, pos):
        px, py = (pos[0] + 0.5) * self.GRID_SIZE, (pos[1] + 0.5) * self.GRID_SIZE
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vx, vy = math.cos(angle) * speed, math.sin(angle) * speed
            lifetime = self.np_random.integers(15, 30)
            self.particles.append([px, py, vx, vy, lifetime])

    def _render_particles(self):
        active_particles = []
        for p in self.particles:
            p[0] += p[2]
            p[1] += p[3]
            p[4] -= 1
            if p[4] > 0:
                radius = int(max(0, p[4] / 6))
                if radius > 0:
                    temp_surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
                    pygame.draw.circle(temp_surf, self.COLOR_PARTICLE, (radius, radius), radius)
                    self.screen.blit(temp_surf, (int(p[0]-radius), int(p[1]-radius)))
                active_particles.append(p)
        self.particles = active_particles

    def _render_ui(self):
        p1_surf = self.font_small.render(f"Player: {len(self.player_body)}", True, self.COLOR_PLAYER)
        self.screen.blit(p1_surf, (10, 10))

        p2_surf = self.font_small.render(f"Opponent: {len(self.opponent_body)}", True, self.COLOR_OPPONENT)
        p2_rect = p2_surf.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(p2_surf, p2_rect)
        
        turns_surf = self.font_small.render(f"Turns Left: {max(0, self.MAX_STEPS - self.steps)}", True, self.COLOR_TEXT)
        turns_rect = turns_surf.get_rect(midtop=(self.WIDTH // 2, 10))
        self.screen.blit(turns_surf, turns_rect)
        
        if self.game_over:
            winner_text, color = "", self.COLOR_TEXT
            player_head = self.player_body[0]
            if not (0 <= player_head[0] < self.GRID_WIDTH and 0 <= player_head[1] < self.GRID_HEIGHT):
                winner_text, color = "WALL COLLISION", self.COLOR_OPPONENT
            elif player_head in list(self.player_body)[1:]:
                winner_text, color = "SELF COLLISION", self.COLOR_OPPONENT
            elif player_head in self.opponent_body:
                winner_text, color = "OPPONENT COLLISION", self.COLOR_OPPONENT
            elif len(self.player_body) > len(self.opponent_body):
                winner_text, color = "YOU WIN!", self.COLOR_PLAYER
            elif len(self.player_body) < len(self.opponent_body):
                winner_text, color = "YOU LOSE", self.COLOR_OPPONENT
            else:
                 winner_text, color = "DRAW", self.COLOR_TEXT

            end_surf = self.font_large.render(winner_text, True, color)
            end_rect = end_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(end_surf, end_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_length": len(self.player_body),
            "opponent_length": len(self.opponent_body),
            "food_pos": self.food_pos,
        }

    def _manhattan_distance(self, p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def close(self):
        pygame.font.quit()
        pygame.quit()