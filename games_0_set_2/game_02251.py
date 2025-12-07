
# Generated: 2025-08-28T04:14:06.388739
# Source Brief: brief_02251.md
# Brief Index: 2251

        
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


class Particle:
    """A simple particle class for visual effects."""
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(1, 4)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.life = random.randint(15, 30)  # Lifetime in frames
        self.radius = random.uniform(2, 5)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.life -= 1
        self.radius -= 0.1
        return self.life > 0 and self.radius > 0

    def draw(self, surface):
        if self.life > 0 and self.radius > 0:
            pos = (int(self.x), int(self.y))
            pygame.gfxdraw.filled_circle(surface, pos[0], pos[1], int(self.radius), self.color)
            pygame.gfxdraw.aacircle(surface, pos[0], pos[1], int(self.radius), self.color)

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # User-facing strings
    user_guide = "Controls: Arrow keys to move. Push boxes (brown) onto targets (red X)."
    game_description = "Push all boxes onto the targets before the timer runs out in this fast-paced puzzle game."
    
    # Frame advance setting
    auto_advance = True

    # --- Constants ---
    # Screen and Grid
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 12, 10
    TILE_SIZE = 40
    GRID_AREA_WIDTH = GRID_WIDTH * TILE_SIZE
    GRID_AREA_HEIGHT = GRID_HEIGHT * TILE_SIZE
    X_OFFSET = (SCREEN_WIDTH - GRID_AREA_WIDTH) // 2
    Y_OFFSET = (SCREEN_HEIGHT - GRID_AREA_HEIGHT) // 2

    # Colors
    COLOR_BG = (25, 28, 36)
    COLOR_GRID = (40, 44, 52)
    COLOR_WALL = (60, 65, 75)
    COLOR_PLAYER = (97, 212, 138)
    COLOR_PLAYER_GLOW = (97, 212, 138, 50)
    COLOR_BOX = (199, 146, 92)
    COLOR_BOX_ON_TARGET = (180, 220, 150)
    COLOR_TARGET = (224, 108, 117)
    COLOR_TEXT = (200, 205, 215)
    COLOR_TIMER = (229, 192, 123)
    COLOR_PARTICLE = (255, 220, 150)

    # Game parameters
    FPS = 30
    MAX_STEPS = 1500
    INITIAL_TIME = 30.0
    NUM_BOXES = 3
    REVERSE_MOVES = 30
    MOVE_COOLDOWN_FRAMES = 6

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
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
        self.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_game_over = pygame.font.SysFont("Consolas", 48, bold=True)
        
        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_state = False
        self.timer = 0.0
        self.player_pos = [0, 0]
        self.boxes = []
        self.targets = []
        self.walls = set()
        self.particles = []
        self.move_cooldown = 0
        self.box_distances = {} # {box_idx: min_dist_to_target}

        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_state = False
        self.timer = self.INITIAL_TIME
        self.particles.clear()
        self.move_cooldown = 0
        
        self._generate_level()
        self.box_distances = self._get_all_box_distances()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = -0.01  # Small penalty for time passing

        if not self.game_over:
            # Unpack action
            movement = action[0]
            
            # Update timers
            self.timer = max(0, self.timer - 1.0 / self.FPS)
            self.move_cooldown = max(0, self.move_cooldown - 1)

            # Update particles
            self.particles = [p for p in self.particles if p.update()]

            # Handle player movement
            if movement != 0 and self.move_cooldown == 0:
                reward += self._handle_movement(movement)
                self.move_cooldown = self.MOVE_COOLDOWN_FRAMES

            # Check termination conditions
            on_target_count = self._count_boxes_on_targets()
            self.score = on_target_count
            
            if on_target_count == self.NUM_BOXES:
                self.game_over = True
                self.win_state = True
                reward += 100.0  # Win reward
            elif self.timer <= 0:
                self.game_over = True
                self.win_state = False
                reward += -50.0 # Loss reward
            
        if self.steps >= self.MAX_STEPS:
            self.game_over = True

        terminated = self.game_over
        self.steps += 1
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_movement(self, movement):
        reward = 0.0
        dx, dy = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}[movement]
        
        next_pos = (self.player_pos[0] + dx, self.player_pos[1] + dy)
        
        # Wall collision
        if next_pos in self.walls:
            return 0.0

        # Box collision
        box_idx = self._get_box_at(next_pos)
        if box_idx is not None:
            box_next_pos = (next_pos[0] + dx, next_pos[1] + dy)
            if box_next_pos in self.walls or self._get_box_at(box_next_pos) is not None:
                return 0.0  # Can't push box
            
            # Move box
            old_box_pos = self.boxes[box_idx]
            old_dist = self._get_min_dist_to_target(old_box_pos)
            
            self.boxes[box_idx] = box_next_pos
            new_dist = self._get_min_dist_to_target(box_next_pos)

            # Reward for box movement
            if new_dist < old_dist:
                reward += 0.1
            elif new_dist > old_dist:
                reward -= 0.2
            
            # Reward for placing on target
            if box_next_pos in self.targets and old_box_pos not in self.targets:
                reward += 10.0
                self._spawn_particles(box_next_pos)
                # Sound: sfx_box_on_target.wav
        
        # Move player
        self.player_pos[0] = next_pos[0]
        self.player_pos[1] = next_pos[1]
        return reward

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid lines
        for i in range(self.GRID_WIDTH + 1):
            x = self.X_OFFSET + i * self.TILE_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.Y_OFFSET), (x, self.Y_OFFSET + self.GRID_AREA_HEIGHT))
        for i in range(self.GRID_HEIGHT + 1):
            y = self.Y_OFFSET + i * self.TILE_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.X_OFFSET, y), (self.X_OFFSET + self.GRID_AREA_WIDTH, y))

        # Draw walls
        for wx, wy in self.walls:
            rect = pygame.Rect(self.X_OFFSET + wx * self.TILE_SIZE, self.Y_OFFSET + wy * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_WALL, rect)

        # Draw targets
        for tx, ty in self.targets:
            center_x = self.X_OFFSET + tx * self.TILE_SIZE + self.TILE_SIZE // 2
            center_y = self.Y_OFFSET + ty * self.TILE_SIZE + self.TILE_SIZE // 2
            size = self.TILE_SIZE // 3
            pygame.gfxdraw.line(self.screen, center_x - size, center_y - size, center_x + size, center_y + size, self.COLOR_TARGET)
            pygame.gfxdraw.line(self.screen, center_x - size, center_y + size, center_x + size, center_y - size, self.COLOR_TARGET)

        # Draw boxes
        for i, (bx, by) in enumerate(self.boxes):
            rect = pygame.Rect(self.X_OFFSET + bx * self.TILE_SIZE + 4, self.Y_OFFSET + by * self.TILE_SIZE + 4, self.TILE_SIZE - 8, self.TILE_SIZE - 8)
            color = self.COLOR_BOX_ON_TARGET if (bx, by) in self.targets else self.COLOR_BOX
            pygame.draw.rect(self.screen, color, rect, border_radius=4)
        
        # Draw player
        px, py = self.player_pos
        player_center_x = self.X_OFFSET + px * self.TILE_SIZE + self.TILE_SIZE // 2
        player_center_y = self.Y_OFFSET + py * self.TILE_SIZE + self.TILE_SIZE // 2
        
        # Glow effect
        glow_radius = int(self.TILE_SIZE * 0.6)
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(glow_surf, glow_radius, glow_radius, glow_radius, self.COLOR_PLAYER_GLOW)
        self.screen.blit(glow_surf, (player_center_x - glow_radius, player_center_y - glow_radius))

        player_rect = pygame.Rect(self.X_OFFSET + px * self.TILE_SIZE + 6, self.Y_OFFSET + py * self.TILE_SIZE + 6, self.TILE_SIZE - 12, self.TILE_SIZE - 12)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=3)
        
        # Draw particles
        for p in self.particles:
            p.draw(self.screen)

    def _render_ui(self):
        # Score display
        score_text = f"SCORE: {self.score}/{self.NUM_BOXES}"
        score_surf = self.font_ui.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (20, 10))

        # Timer display
        timer_text = f"TIME: {self.timer:.1f}"
        timer_surf = self.font_ui.render(timer_text, True, self.COLOR_TIMER)
        self.screen.blit(timer_surf, (self.SCREEN_WIDTH - timer_surf.get_width() - 20, 10))

        # Game Over display
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg = "LEVEL CLEAR!" if self.win_state else "TIME UP!"
            color = self.COLOR_PLAYER if self.win_state else self.COLOR_TARGET
            msg_surf = self.font_game_over.render(msg, True, color)
            msg_rect = msg_surf.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
            self.screen.blit(msg_surf, msg_rect)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _generate_level(self):
        self.walls = set()
        for x in range(self.GRID_WIDTH):
            self.walls.add((x, 0))
            self.walls.add((x, self.GRID_HEIGHT - 1))
        for y in range(self.GRID_HEIGHT):
            self.walls.add((0, y))
            self.walls.add((self.GRID_WIDTH - 1, y))

        playable_coords = []
        for x in range(1, self.GRID_WIDTH - 1):
            for y in range(1, self.GRID_HEIGHT - 1):
                playable_coords.append((x, y))
        
        self.np_random.shuffle(playable_coords)
        
        self.targets = [playable_coords.pop() for _ in range(self.NUM_BOXES)]
        self.boxes = list(self.targets)
        self.player_pos = list(playable_coords.pop())

        # Reverse moves to generate puzzle
        for _ in range(self.REVERSE_MOVES):
            box_idx = self.np_random.integers(0, len(self.boxes))
            box_pos = self.boxes[box_idx]
            
            # Try to "un-push" the box
            direction = random.choice([(0, -1), (0, 1), (-1, 0), (1, 0)])
            player_new_pos = box_pos
            box_new_pos = (box_pos[0] + direction[0], box_pos[1] + direction[1])

            occupied_spaces = set(self.boxes) | self.walls
            
            if box_new_pos not in occupied_spaces:
                self.player_pos = list(player_new_pos)
                self.boxes[box_idx] = box_new_pos

    def _get_box_at(self, pos):
        try:
            return self.boxes.index(pos)
        except ValueError:
            return None

    def _count_boxes_on_targets(self):
        return sum(1 for box in self.boxes if box in self.targets)
    
    @staticmethod
    def _manhattan_distance(p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def _get_min_dist_to_target(self, pos):
        if not self.targets: return 0
        return min(self._manhattan_distance(pos, t) for t in self.targets)

    def _get_all_box_distances(self):
        return {i: self._get_min_dist_to_target(box_pos) for i, box_pos in enumerate(self.boxes)}

    def _spawn_particles(self, pos):
        center_x = self.X_OFFSET + pos[0] * self.TILE_SIZE + self.TILE_SIZE // 2
        center_y = self.Y_OFFSET + pos[1] * self.TILE_SIZE + self.TILE_SIZE // 2
        for _ in range(30):
            self.particles.append(Particle(center_x, center_y, self.COLOR_PARTICLE))

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space
        # Temporarily generate a level to get an observation
        original_state = {
            "player_pos": self.player_pos, "boxes": self.boxes, 
            "targets": self.targets, "walls": self.walls
        }
        if not self.targets: self._generate_level()
        test_obs = self._get_observation()
        # Restore state
        self.player_pos, self.boxes, self.targets, self.walls = original_state.values()

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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Game loop
    running = True
    while running:
        # Pygame event handling
        action = np.array([0, 0, 0]) # Default no-op
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        if keys[pygame.K_SPACE]:
            action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1

        if keys[pygame.K_r]: # Press 'r' to reset
            obs, info = env.reset()
            done = False

        # Step the environment
        if not done:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        # Render the observation to the screen
        real_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        real_screen.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(env.FPS)

    env.close()