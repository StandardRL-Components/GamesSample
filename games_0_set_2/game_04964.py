
# Generated: 2025-08-28T03:34:38.315858
# Source Brief: brief_04964.md
# Brief Index: 4964

        
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
    """A simple particle for visual effects."""
    def __init__(self, x, y, color, life, speed, angle_range):
        self.x = x
        self.y = y
        self.color = color
        self.life = life
        self.max_life = life
        angle = random.uniform(angle_range[0], angle_range[1])
        self.vx = math.cos(angle) * speed * random.uniform(0.5, 1.5)
        self.vy = math.sin(angle) * speed * random.uniform(0.5, 1.5)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.life -= 1

    def draw(self, surface):
        if self.life > 0:
            alpha = int(255 * (self.life / self.max_life))
            radius = int(3 * (self.life / self.max_life))
            if radius > 0:
                # Use gfxdraw for alpha blending
                pygame.gfxdraw.filled_circle(surface, int(self.x), int(self.y), radius, (*self.color, alpha))

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = "Controls: Use arrow keys to move the selector. Press space to move to the selected square."

    # Must be a short, user-facing description of the game:
    game_description = "Navigate a grid to collect all the green gems while avoiding the red traps."

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    # --- Constants ---
    SCREEN_W, SCREEN_H = 640, 400
    GRID_W, GRID_H = 20, 12
    CELL_SIZE = 30
    GRID_OFFSET_X = (SCREEN_W - GRID_W * CELL_SIZE) // 2
    GRID_OFFSET_Y = (SCREEN_H - GRID_H * CELL_SIZE) // 2 + 10 # Shift down for UI

    NUM_GEMS = 50
    NUM_TRAPS = 10
    MAX_STEPS = 1000

    # --- Colors ---
    COLOR_BG = (20, 20, 30)
    COLOR_GRID = (50, 50, 70)
    COLOR_GEM = (0, 255, 128)
    COLOR_GEM_SHINE = (180, 255, 220)
    COLOR_TRAP = (255, 50, 50)
    COLOR_TRAP_X = (150, 0, 0)
    COLOR_PLAYER = (50, 150, 255)
    COLOR_SELECTOR = (255, 255, 0)
    COLOR_TEXT = (220, 220, 220)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_W, self.SCREEN_H))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_big = pygame.font.SysFont("monospace", 48, bold=True)
        
        # Initialize state variables
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_state = ""
        self.space_was_held = False
        self.particles = []

        # Generate all possible grid positions
        all_pos = [(x, y) for x in range(self.GRID_W) for y in range(self.GRID_H)]
        self.np_random.shuffle(all_pos)

        # Place traps and gems
        self.traps = set(all_pos[:self.NUM_TRAPS])
        self.gems = set(all_pos[self.NUM_TRAPS : self.NUM_TRAPS + self.NUM_GEMS])
        
        # Place player in a safe spot
        safe_spots = [pos for pos in all_pos if pos not in self.traps and pos not in self.gems]
        self.player_pos = list(safe_spots[0]) if safe_spots else [0,0]

        self.selector_pos = list(self.player_pos)
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        
        reward = 0.0

        # --- Update selector position ---
        old_selector_pos = tuple(self.selector_pos)
        if movement == 1: self.selector_pos[1] -= 1  # Up
        elif movement == 2: self.selector_pos[1] += 1  # Down
        elif movement == 3: self.selector_pos[0] -= 1  # Left
        elif movement == 4: self.selector_pos[0] += 1  # Right
        
        # Wrap around grid
        self.selector_pos[0] %= self.GRID_W
        self.selector_pos[1] %= self.GRID_H
        new_selector_pos = tuple(self.selector_pos)

        # --- Continuous Reward Shaping for Selector Movement ---
        if movement != 0:
            if self.gems:
                old_dist_gem = self._find_closest_dist(old_selector_pos, self.gems)
                new_dist_gem = self._find_closest_dist(new_selector_pos, self.gems)
                reward += (old_dist_gem - new_dist_gem) * 0.1
            if self.traps:
                old_dist_trap = self._find_closest_dist(old_selector_pos, self.traps)
                new_dist_trap = self._find_closest_dist(new_selector_pos, self.traps)
                reward -= (old_dist_trap - new_dist_trap) * 0.1 # Penalize getting closer

        # --- Handle "Click" Action (on space press) ---
        space_pressed = space_held and not self.space_was_held
        if space_pressed:
            self.player_pos = list(self.selector_pos)
            clicked_pos = tuple(self.player_pos)

            if clicked_pos in self.traps:
                reward += -100
                self.game_over = True
                self.win_state = "LOSE"
                self._create_particles(clicked_pos, self.COLOR_TRAP, 30, 4, (0, 2 * math.pi))
                # Sound: Explosion
            elif clicked_pos in self.gems:
                reward += 10
                self.score += 1
                self.gems.remove(clicked_pos)
                self._create_particles(clicked_pos, self.COLOR_GEM, 20, 3, (-math.pi, 0))
                # Sound: Gem collect
                if not self.gems:
                    reward += 100
                    self.game_over = True
                    self.win_state = "WIN"
                    self._create_particles(clicked_pos, (255,215,0), 50, 5, (0, 2*math.pi))
                    # Sound: Victory
            else: # Clicked an empty square
                reward -= 0.01 # Small penalty for a wasted action

        self.space_was_held = space_held
        
        self.steps += 1
        self._update_particles()
        
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        if terminated and not self.game_over:
            self.win_state = "TIME UP"
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _find_closest_dist(self, pos, targets):
        if not targets: return 0
        px, py = pos
        return min(abs(px - tx) + abs(py - ty) for tx, ty in targets)

    def _create_particles(self, grid_pos, color, count, speed, angle_range):
        px = self.GRID_OFFSET_X + grid_pos[0] * self.CELL_SIZE + self.CELL_SIZE / 2
        py = self.GRID_OFFSET_Y + grid_pos[1] * self.CELL_SIZE + self.CELL_SIZE / 2
        for _ in range(count):
            self.particles.append(Particle(px, py, color, random.randint(20, 40), speed, angle_range))
    
    def _update_particles(self):
        self.particles = [p for p in self.particles if p.life > 0]
        for p in self.particles:
            p.update()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid lines
        for x in range(self.GRID_W + 1):
            start = (self.GRID_OFFSET_X + x * self.CELL_SIZE, self.GRID_OFFSET_Y)
            end = (self.GRID_OFFSET_X + x * self.CELL_SIZE, self.GRID_OFFSET_Y + self.GRID_H * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end)
        for y in range(self.GRID_H + 1):
            start = (self.GRID_OFFSET_X, self.GRID_OFFSET_Y + y * self.CELL_SIZE)
            end = (self.GRID_OFFSET_X + self.GRID_W * self.CELL_SIZE, self.GRID_OFFSET_Y + y * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end)

        # Draw traps
        for x, y in self.traps:
            rect = pygame.Rect(self.GRID_OFFSET_X + x * self.CELL_SIZE, self.GRID_OFFSET_Y + y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_TRAP, rect.inflate(-4, -4))
            pygame.draw.line(self.screen, self.COLOR_TRAP_X, rect.topleft, rect.bottomright, 2)
            pygame.draw.line(self.screen, self.COLOR_TRAP_X, rect.topright, rect.bottomleft, 2)

        # Draw gems (with pulsating shine)
        pulse = (math.sin(self.steps * 0.1) + 1) / 2
        gem_r = int(self.CELL_SIZE * 0.35)
        shine_r = int(gem_r * 0.5 * (1 + pulse * 0.5))
        for x, y in self.gems:
            cx = int(self.GRID_OFFSET_X + x * self.CELL_SIZE + self.CELL_SIZE / 2)
            cy = int(self.GRID_OFFSET_Y + y * self.CELL_SIZE + self.CELL_SIZE / 2)
            pygame.gfxdraw.filled_circle(self.screen, cx, cy, gem_r, self.COLOR_GEM)
            pygame.gfxdraw.filled_circle(self.screen, cx, cy, shine_r, self.COLOR_GEM_SHINE)

        # Draw player
        player_rect = pygame.Rect(self.GRID_OFFSET_X + self.player_pos[0] * self.CELL_SIZE, self.GRID_OFFSET_Y + self.player_pos[1] * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect.inflate(-4, -4))
        
        # Draw selector
        if not self.game_over:
            selector_rect = pygame.Rect(self.GRID_OFFSET_X + self.selector_pos[0] * self.CELL_SIZE, self.GRID_OFFSET_Y + self.selector_pos[1] * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_SELECTOR, selector_rect, 3)
            
        for p in self.particles: p.draw(self.screen)

    def _render_ui(self):
        score_text = self.font_main.render(f"GEMS: {self.score}/{self.NUM_GEMS}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (15, 10))
        steps_text = self.font_main.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (self.SCREEN_W - steps_text.get_width() - 15, 10))
        
        if self.game_over:
            msg, color = "", self.COLOR_TEXT
            if self.win_state == "WIN": msg, color = "YOU WIN!", self.COLOR_GEM
            elif self.win_state == "LOSE": msg, color = "GAME OVER", self.COLOR_TRAP
            elif self.win_state == "TIME UP": msg, color = "TIME'S UP!", self.COLOR_SELECTOR
            
            end_text = self.font_big.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_W / 2, self.SCREEN_H / 2))
            bg_rect = text_rect.inflate(20, 20)
            s = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
            s.fill((0,0,0,180))
            self.screen.blit(s, bg_rect.topleft)
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def close(self):
        pygame.quit()

# --- Example of how to run the environment for manual play ---
if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    
    try:
        pygame.display.init()
        pygame.font.init()
        screen = pygame.display.set_mode((env.SCREEN_W, env.SCREEN_H))
        pygame.display.set_caption("Gem Grid")
        clock = pygame.time.Clock()

        obs, info = env.reset()
        done = False
        
        print("\n--- Manual Game Start ---")
        print(env.game_description)
        print(env.user_guide)

        while not done:
            movement, space_held = 0, False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True

            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            if keys[pygame.K_SPACE]: space_held = True

            action = [movement, 1 if space_held else 0, 0]
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if reward != 0:
                print(f"Step: {info['steps']}, Score: {info['score']}, Reward: {reward:.2f}")

            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            clock.tick(15) # Lower tick rate for responsive turn-based play

        print(f"Game Over! Final Score: {info['score']}")
        pygame.time.wait(2000)

    finally:
        env.close()
        pygame.quit()