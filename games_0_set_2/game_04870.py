
# Generated: 2025-08-28T03:16:24.148408
# Source Brief: brief_04870.md
# Brief Index: 4870

        
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


# Particle class for visual effects
class Particle:
    def __init__(self, x, y, color, life, size_range=(2, 5), speed_range=(1, 3)):
        self.x = x
        self.y = y
        self.color = color
        self.life = life
        self.max_life = life
        self.size = random.uniform(size_range[0], size_range[1])
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(speed_range[0], speed_range[1])
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed

    def update(self):
        self.life -= 1
        self.x += self.vx
        self.y += self.vy
        self.vx *= 0.98  # friction
        self.vy *= 0.98

    def draw(self, surface):
        if self.life > 0:
            alpha = int(255 * (self.life / self.max_life))
            temp_surface = pygame.Surface((self.size * 2, self.size * 2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surface, (*self.color, alpha), (self.size, self.size), self.size)
            surface.blit(temp_surface, (self.x - self.size, self.y - self.size), special_flags=pygame.BLEND_RGBA_ADD)

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move. Push crates onto the red 'X' targets before time runs out."
    )

    game_description = (
        "A fast-paced puzzle game. Race against the clock to push all crates onto their targets."
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 16, 10
    CELL_SIZE = 40
    
    NUM_CRATES = 7
    TIME_LIMIT_SECONDS = 90
    MAX_STEPS = 2700  # 90 seconds * 30 FPS

    COLOR_BG = (15, 20, 30)
    COLOR_GRID = (30, 40, 60)
    
    COLOR_PLAYER = (50, 220, 50)
    COLOR_PLAYER_SHADOW = (30, 150, 30)

    COLOR_CRATE = (160, 110, 70)
    COLOR_CRATE_SHADOW = (110, 75, 50)
    COLOR_CRATE_ON_TARGET = (200, 180, 50)

    COLOR_TARGET = (220, 40, 40)
    COLOR_UI_TEXT = (240, 240, 240)
    COLOR_TIMER_WARN = (255, 200, 0)
    COLOR_TIMER_CRITICAL = (255, 50, 50)

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
        self.font_large = pygame.font.SysFont("Consolas", 30, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_x = pygame.font.SysFont("Arial", 30, bold=True)

        self.player_pos = (0, 0)
        self.crates = []
        self.targets = []
        self.walls = set()
        self.particles = []
        self.crates_on_target = 0
        self.last_dist_to_crate = 0

        self.steps = 0
        self.score = 0
        self.time_left = 0
        self.game_over = False
        
        self.np_random = None

        self.reset()
        # self.validate_implementation() # Commented out for final submission

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
             self.np_random = np.random.default_rng(seed=seed)
        else:
             self.np_random = np.random.default_rng()

        self._generate_level()
        
        self.steps = 0
        self.score = 0
        self.time_left = self.TIME_LIMIT_SECONDS * 30  # 30 FPS
        self.game_over = False
        self.particles = []
        
        self.last_dist_to_crate = self._get_dist_to_nearest_off_target_crate()

        return self._get_observation(), self._get_info()

    def _generate_level(self):
        self.walls.clear()
        self.crates.clear()
        self.targets.clear()

        # Add perimeter walls
        for x in range(self.GRID_WIDTH):
            self.walls.add((x, -1))
            self.walls.add((x, self.GRID_HEIGHT))
        for y in range(self.GRID_HEIGHT):
            self.walls.add((-1, y))
            self.walls.add((self.GRID_WIDTH, y))

        # Add some internal walls
        for _ in range(self.np_random.integers(3, 6)):
            wall_len = self.np_random.integers(2, 5)
            start_x = self.np_random.integers(1, self.GRID_WIDTH - wall_len - 1)
            start_y = self.np_random.integers(1, self.GRID_HEIGHT - wall_len - 1)
            if self.np_random.random() > 0.5: # Horizontal
                for i in range(wall_len):
                    self.walls.add((start_x + i, start_y))
            else: # Vertical
                for i in range(wall_len):
                    self.walls.add((start_x, start_y + i))

        # Generate valid positions
        valid_pos = []
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                if (x, y) not in self.walls:
                    valid_pos.append((x, y))

        # Ensure enough space for targets, crates and player
        if len(valid_pos) < self.NUM_CRATES * 2 + 1:
            # Fallback to a less cluttered map if generation fails
            return self.reset()

        # Place targets
        target_indices = self.np_random.choice(len(valid_pos), size=self.NUM_CRATES, replace=False)
        self.targets = [valid_pos[i] for i in target_indices]
        
        # Place crates on targets (initial solved state)
        self.crates = list(self.targets)

        # Scramble puzzle by reverse-pushing crates
        for _ in range(50): # Number of reverse moves
            crate_idx = self.np_random.integers(0, len(self.crates))
            crate_pos = self.crates[crate_idx]
            
            # Choose a direction to "pull" from
            # 0:up, 1:down, 2:left, 3:right
            move_dir = self.np_random.integers(0, 4)
            if move_dir == 0: dx, dy = 0, -1
            elif move_dir == 1: dx, dy = 0, 1
            elif move_dir == 2: dx, dy = -1, 0
            else: dx, dy = 1, 0

            # The spot the crate will move to
            new_crate_pos = (crate_pos[0] + dx, crate_pos[1] + dy)
            # The spot the player needs to be in to perform the original push
            player_stand_pos = (crate_pos[0] - dx, crate_pos[1] - dy)

            # Check if reverse move is valid
            if (new_crate_pos not in self.walls and
                new_crate_pos not in self.crates and
                player_stand_pos not in self.walls):
                self.crates[crate_idx] = new_crate_pos
        
        # Place player in a random empty spot
        empty_pos = [p for p in valid_pos if p not in self.crates]
        if not empty_pos:
            return self.reset() # Should not happen, but as a safeguard
        self.player_pos = empty_pos[self.np_random.integers(0, len(empty_pos))]


    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        self.time_left -= 1
        self.steps += 1

        movement = action[0]
        px, py = self.player_pos
        nx, ny = px, py

        if movement == 1: ny -= 1 # Up
        elif movement == 2: ny += 1 # Down
        elif movement == 3: nx -= 1 # Left
        elif movement == 4: nx += 1 # Right

        if (nx, ny) != (px, py): # If a move was attempted
            # Check for wall collision
            if (nx, ny) in self.walls:
                pass # Player hits a wall, no move
            # Check for crate collision
            elif (nx, ny) in self.crates:
                crate_idx = self.crates.index((nx, ny))
                # Position behind the crate
                bx, by = nx + (nx - px), ny + (ny - py)
                # Check if space behind crate is free
                if (bx, by) not in self.walls and (bx, by) not in self.crates:
                    # Push crate
                    self.crates[crate_idx] = (bx, by)
                    self.player_pos = (nx, ny)
                    # # Sound: crate_push.wav
                    
                    # Check if crate landed on a target
                    if (bx, by) in self.targets and (nx, ny) not in self.targets:
                        reward += 10
                        self._create_particles(bx, by, self.COLOR_CRATE_ON_TARGET)
                        # # Sound: success_chime.wav
            else:
                # No collision, move player
                self.player_pos = (nx, ny)
        
        # Calculate distance-based reward
        dist_now = self._get_dist_to_nearest_off_target_crate()
        if dist_now < self.last_dist_to_crate:
            reward += 0.1
        elif dist_now > self.last_dist_to_crate:
            reward -= 0.1
        self.last_dist_to_crate = dist_now

        self.score += reward
        
        # Update particles
        for p in self.particles:
            p.update()
        self.particles = [p for p in self.particles if p.life > 0]

        # Check termination conditions
        terminated = self._check_termination()
        if terminated:
            self.game_over = True
            if self.crates_on_target == self.NUM_CRATES:
                reward += 100 # Win bonus
                self.score += 100
            else:
                reward -= 100 # Lose penalty
                self.score -= 100

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _get_dist_to_nearest_off_target_crate(self):
        off_target_crates = [c for c in self.crates if c not in self.targets]
        if not off_target_crates:
            return 0
        
        px, py = self.player_pos
        min_dist = float('inf')
        for cx, cy in off_target_crates:
            dist = abs(px - cx) + abs(py - cy) - 1 # -1 because player stands next to crate
            if dist < min_dist:
                min_dist = dist
        return max(0, min_dist)

    def _check_termination(self):
        self.crates_on_target = sum(1 for c in self.crates if c in self.targets)
        if self.crates_on_target == self.NUM_CRATES:
            return True
        if self.time_left <= 0:
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        return False

    def _create_particles(self, grid_x, grid_y, color):
        px = (grid_x + 0.5) * self.CELL_SIZE
        py = (grid_y + 0.5) * self.CELL_SIZE
        for _ in range(30):
            self.particles.append(Particle(px, py, color, life=random.randint(20, 40)))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for x in range(0, self.SCREEN_WIDTH, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

        # Draw targets
        target_text = self.font_x.render("X", True, self.COLOR_TARGET)
        for tx, ty in self.targets:
            rect = target_text.get_rect(center=((tx + 0.5) * self.CELL_SIZE, (ty + 0.5) * self.CELL_SIZE))
            self.screen.blit(target_text, rect)
            
        # Draw crates
        for i, (cx, cy) in enumerate(self.crates):
            is_on_target = (cx, cy) in self.targets
            main_color = self.COLOR_CRATE_ON_TARGET if is_on_target else self.COLOR_CRATE
            shadow_color = self.COLOR_CRATE_SHADOW # Shadow is same regardless
            
            px, py = cx * self.CELL_SIZE, cy * self.CELL_SIZE
            inset = 4
            shadow_offset = 3
            # Shadow
            pygame.draw.rect(self.screen, shadow_color, (px + inset, py + inset, self.CELL_SIZE - inset*2, self.CELL_SIZE - inset*2))
            # Main body
            pygame.draw.rect(self.screen, main_color, (px + inset, py + inset, self.CELL_SIZE - inset*2 - shadow_offset, self.CELL_SIZE - inset*2 - shadow_offset))

        # Draw player
        px, py = self.player_pos
        screen_x, screen_y = px * self.CELL_SIZE, py * self.CELL_SIZE
        inset = 6
        shadow_offset = 3
        # Shadow
        pygame.draw.rect(self.screen, self.COLOR_PLAYER_SHADOW, (screen_x + inset, screen_y + inset, self.CELL_SIZE - inset*2, self.CELL_SIZE - inset*2))
        # Main Body
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, (screen_x + inset, screen_y + inset, self.CELL_SIZE - inset*2 - shadow_offset, self.CELL_SIZE - inset*2 - shadow_offset))

        # Draw particles
        for p in self.particles:
            p.draw(self.screen)

    def _render_ui(self):
        # Draw crates on target count
        crates_text = f"COMPLETED: {self.crates_on_target}/{self.NUM_CRATES}"
        text_surf = self.font_small.render(crates_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(text_surf, (10, 10))
        
        # Draw timer
        seconds_left = max(0, self.time_left / 30)
        timer_text = f"TIME: {seconds_left:.1f}"
        
        timer_color = self.COLOR_UI_TEXT
        if seconds_left < 30: timer_color = self.COLOR_TIMER_WARN
        if seconds_left < 10: timer_color = self.COLOR_TIMER_CRITICAL
            
        text_surf = self.font_small.render(timer_text, True, timer_color)
        text_rect = text_surf.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(text_surf, text_rect)

        # Draw game over message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            
            if self.crates_on_target == self.NUM_CRATES:
                msg = "PUZZLE COMPLETE!"
                color = self.COLOR_CRATE_ON_TARGET
            else:
                msg = "TIME UP!"
                color = self.COLOR_TIMER_CRITICAL
            
            text_surf = self.font_large.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            overlay.blit(text_surf, text_rect)
            self.screen.blit(overlay, (0, 0))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left_seconds": round(self.time_left / 30, 2),
            "crates_on_target": self.crates_on_target
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    done = False
    total_reward = 0
    
    # --- Pygame setup for human play ---
    render_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Sokoban Runner")
    game_clock = pygame.time.Clock()
    
    action = env.action_space.sample()
    action.fill(0) # Start with no-op

    print(env.user_guide)

    while not done:
        # --- Human Controls ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        keys = pygame.key.get_pressed()
        movement = 0 # No-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        action[0] = movement
        # space/shift are not used in this game
        action[1] = 1 if keys[pygame.K_SPACE] else 0
        action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        # --- Render to screen ---
        # The observation is (H, W, C), but pygame blit needs a Surface.
        # We can get the surface from the env directly before it's converted.
        surf = pygame.transform.flip(env.screen, False, True)
        surf = pygame.transform.rotate(surf, -90)
        render_screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        game_clock.tick(30) # Match env's internal clock

    print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
    
    pygame.quit()
    env.close()