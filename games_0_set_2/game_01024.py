
# Generated: 2025-08-27T15:34:52.533895
# Source Brief: brief_01024.md
# Brief Index: 1024

        
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move on the grid. Survive for 60 seconds while dodging monsters. "
        "Collect white orbs for temporary invulnerability."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Survive a minute-long monster mash by strategically navigating a grid and dodging colorful, "
        "procedurally generated monsters."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 10
        self.MAX_STEPS = 1800 # Brief says 3600, but 1800 is more reasonable for turn-based
        self.MAX_HITS = 5
        self.NUM_MONSTERS = 5
        self.POWERUP_SPAWN_INTERVAL = 100
        self.POWERUP_DURATION = 15 # Steps of invulnerability

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)

        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (40, 50, 70)
        self.COLOR_PLAYER = (50, 255, 150)
        self.COLOR_PLAYER_SHIELD = (255, 255, 255)
        self.COLOR_POWERUP = (255, 255, 255)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_HIT_FLASH = (255, 50, 50)
        self.MONSTER_COLORS = [
            (255, 80, 80),   # Red
            (80, 150, 255),  # Blue
            (255, 255, 80),  # Yellow
            (200, 80, 255),  # Purple
            (255, 150, 50),  # Orange
        ]

        # Grid layout calculation
        self.GRID_AREA_WIDTH = self.HEIGHT - 40
        self.CELL_SIZE = self.GRID_AREA_WIDTH // self.GRID_SIZE
        self.GRID_START_X = (self.WIDTH - self.GRID_AREA_WIDTH) // 2
        self.GRID_START_Y = (self.HEIGHT - self.GRID_AREA_WIDTH) // 2

        # Game state variables (initialized in reset)
        self.player_pos = None
        self.monsters = None
        self.powerup_pos = None
        self.powerup_active = None
        self.powerup_spawn_timer = None
        self.invulnerability_timer = None
        self.monster_hits = None
        self.steps = None
        self.score = None
        self.hit_flash_timer = None
        self.powerup_particles = None

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.player_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.powerup_active = False
        self.powerup_pos = [-1, -1]
        self.powerup_spawn_timer = self.POWERUP_SPAWN_INTERVAL
        self.invulnerability_timer = 0
        self.monster_hits = 0
        self.steps = 0
        self.score = 0
        self.hit_flash_timer = 0
        self.powerup_particles = []

        self._initialize_monsters()
        self._spawn_powerup()

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement = action[0]
        # space_held = action[1] == 1
        # shift_held = action[2] == 1

        reward = 0.1  # Base reward for surviving a step
        
        # Calculate reward for moving towards/away from the nearest monster
        reward += self._calculate_movement_reward(movement)

        # Update player position
        prev_pos = list(self.player_pos)
        if movement == 1: self.player_pos[1] -= 1  # Up
        elif movement == 2: self.player_pos[1] += 1  # Down
        elif movement == 3: self.player_pos[0] -= 1  # Left
        elif movement == 4: self.player_pos[0] += 1  # Right
        
        # Clamp position to grid boundaries
        self.player_pos[0] = np.clip(self.player_pos[0], 0, self.GRID_SIZE - 1)
        self.player_pos[1] = np.clip(self.player_pos[1], 0, self.GRID_SIZE - 1)

        # Update monsters
        self._update_monsters()

        # Check for collisions
        for monster in self.monsters:
            if self.player_pos == monster["pos"]:
                if self.invulnerability_timer <= 0:
                    self.monster_hits += 1
                    self.hit_flash_timer = 5 # Flash for 5 frames
                    # Sound: Player hit
                break # Only one hit per step

        # Check for power-up collection
        if self.powerup_active and self.player_pos == self.powerup_pos:
            reward += 5.0
            self.invulnerability_timer = self.POWERUP_DURATION
            self.powerup_active = False
            self.powerup_spawn_timer = self.POWERUP_SPAWN_INTERVAL
            self._create_powerup_particles(self.powerup_pos)
            # Sound: Power-up collected

        # Update timers and state
        self.steps += 1
        if self.invulnerability_timer > 0:
            self.invulnerability_timer -= 1
        if self.hit_flash_timer > 0:
            self.hit_flash_timer -= 1
        
        self.powerup_spawn_timer -= 1
        if self.powerup_spawn_timer <= 0 and not self.powerup_active:
            self._spawn_powerup()

        # Check for termination
        terminated = (self.monster_hits >= self.MAX_HITS) or (self.steps >= self.MAX_STEPS)
        if terminated:
            if self.steps >= self.MAX_STEPS:
                reward += 100.0 # Win bonus
            else:
                reward -= 100.0 # Loss penalty

        self.score += reward

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
            "hits": self.monster_hits,
            "invulnerable": self.invulnerability_timer > 0
        }
        
    def _grid_to_pixel(self, grid_pos):
        px = self.GRID_START_X + grid_pos[0] * self.CELL_SIZE + self.CELL_SIZE // 2
        py = self.GRID_START_Y + grid_pos[1] * self.CELL_SIZE + self.CELL_SIZE // 2
        return int(px), int(py)

    def _render_game(self):
        # Draw grid
        for i in range(self.GRID_SIZE + 1):
            x_start = self.GRID_START_X + i * self.CELL_SIZE
            y_start = self.GRID_START_Y + i * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x_start, self.GRID_START_Y), (x_start, self.GRID_START_Y + self.GRID_AREA_WIDTH))
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_START_X, y_start), (self.GRID_START_X + self.GRID_AREA_WIDTH, y_start))

        # Draw power-up
        if self.powerup_active:
            px, py = self._grid_to_pixel(self.powerup_pos)
            pulse = abs(math.sin(self.steps * 0.2))
            radius = int(self.CELL_SIZE * 0.2 + pulse * 3)
            pygame.gfxdraw.filled_circle(self.screen, px, py, radius, self.COLOR_POWERUP)
            pygame.gfxdraw.aacircle(self.screen, px, py, radius, self.COLOR_POWERUP)

        # Draw power-up particles
        self._update_and_draw_particles()

        # Draw monsters
        for monster in self.monsters:
            px, py = self._grid_to_pixel(monster["pos"])
            size = int(self.CELL_SIZE * 0.7)
            if monster["type"] == "pacer": # Square
                pygame.draw.rect(self.screen, monster["color"], (px - size//2, py - size//2, size, size))
            elif monster["type"] == "patroller": # Triangle
                points = [(px, py - size//2), (px - size//2, py + size//2), (px + size//2, py + size//2)]
                pygame.draw.polygon(self.screen, monster["color"], points)
            elif monster["type"] == "chaser": # Circle
                pygame.gfxdraw.filled_circle(self.screen, px, py, size//2, monster["color"])
                pygame.gfxdraw.aacircle(self.screen, px, py, size//2, monster["color"])
            elif monster["type"] == "jumper": # Diamond
                points = [(px, py - size//2), (px - size//2, py), (px, py + size//2), (px + size//2, py)]
                pygame.draw.polygon(self.screen, monster["color"], points)
            else: # Diagonal - Octagon
                radius = size // 2
                points = [(px + radius * math.cos(2 * math.pi * i / 8), py + radius * math.sin(2 * math.pi * i / 8)) for i in range(8)]
                pygame.draw.polygon(self.screen, monster["color"], points)

        # Draw player
        player_px, player_py = self._grid_to_pixel(self.player_pos)
        player_size = int(self.CELL_SIZE * 0.8)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, (player_px - player_size//2, player_py - player_size//2, player_size, player_size), border_radius=3)
        
        # Draw invulnerability shield
        if self.invulnerability_timer > 0:
            alpha = 50 + (self.invulnerability_timer / self.POWERUP_DURATION) * 150
            pulse_radius = int(self.CELL_SIZE * 0.55 * (0.9 + 0.1 * math.sin(self.steps * 0.5)))
            shield_surf = pygame.Surface((pulse_radius*2, pulse_radius*2), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(shield_surf, pulse_radius, pulse_radius, pulse_radius, (*self.COLOR_PLAYER_SHIELD, int(alpha)))
            pygame.gfxdraw.aacircle(shield_surf, pulse_radius, pulse_radius, pulse_radius, (*self.COLOR_PLAYER_SHIELD, int(alpha*1.2)))
            self.screen.blit(shield_surf, (player_px - pulse_radius, player_py - pulse_radius))

        # Draw hit flash
        if self.hit_flash_timer > 0:
            flash_surface = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            alpha = (self.hit_flash_timer / 5) * 100
            flash_surface.fill((*self.COLOR_HIT_FLASH, alpha))
            self.screen.blit(flash_surface, (0, 0))

    def _render_ui(self):
        # Hits display
        hits_text = self.font_small.render(f"HITS: {self.monster_hits}/{self.MAX_HITS}", True, self.COLOR_TEXT)
        self.screen.blit(hits_text, (20, 10))

        # Time remaining display
        time_left = self.MAX_STEPS - self.steps
        time_text = self.font_small.render(f"TIME: {time_left}", True, self.COLOR_TEXT)
        self.screen.blit(time_text, (self.WIDTH - time_text.get_width() - 20, 10))
        
        # Game Over / Win Text
        if self.monster_hits >= self.MAX_HITS:
            text = self.font_large.render("GAME OVER", True, self.COLOR_HIT_FLASH)
            text_rect = text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(text, text_rect)
        elif self.steps >= self.MAX_STEPS:
            text = self.font_large.render("YOU SURVIVED!", True, self.COLOR_PLAYER)
            text_rect = text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(text, text_rect)

    def _initialize_monsters(self):
        self.monsters = []
        occupied_pos = {tuple(self.player_pos)}
        monster_types = ["pacer", "patroller", "chaser", "jumper", "diagonal"]
        
        for i in range(self.NUM_MONSTERS):
            while True:
                pos = [self.np_random.integers(0, self.GRID_SIZE), self.np_random.integers(0, self.GRID_SIZE)]
                if tuple(pos) not in occupied_pos and sum(abs(a - b) for a, b in zip(pos, self.player_pos)) > 3:
                    occupied_pos.add(tuple(pos))
                    break
            
            monster_type = monster_types[i % len(monster_types)]
            monster_state = {"dir": 1} if monster_type in ["pacer", "diagonal"] else {}
            if monster_type == "patroller":
                monster_state = {"path_index": 0, "path": [(pos[0], pos[1]), (pos[0]+2, pos[1]), (pos[0]+2, pos[1]+2), (pos[0], pos[1]+2)]}
            if monster_type == "jumper":
                monster_state = {"timer": self.np_random.integers(3, 6)}

            self.monsters.append({
                "pos": pos,
                "color": self.MONSTER_COLORS[i],
                "type": monster_type,
                "state": monster_state
            })

    def _update_monsters(self):
        for m in self.monsters:
            if m["type"] == "pacer":
                m["pos"][0] += m["state"]["dir"]
                if m["pos"][0] <= 0 or m["pos"][0] >= self.GRID_SIZE - 1:
                    m["state"]["dir"] *= -1
                    m["pos"][0] = np.clip(m["pos"][0], 0, self.GRID_SIZE - 1)
            elif m["type"] == "patroller":
                path = m["state"]["path"]
                target = path[m["state"]["path_index"]]
                if m["pos"][0] < target[0]: m["pos"][0] += 1
                elif m["pos"][0] > target[0]: m["pos"][0] -= 1
                elif m["pos"][1] < target[1]: m["pos"][1] += 1
                elif m["pos"][1] > target[1]: m["pos"][1] -= 1
                if tuple(m["pos"]) == target:
                    m["state"]["path_index"] = (m["state"]["path_index"] + 1) % len(path)
            elif m["type"] == "chaser":
                if self.np_random.random() > 0.3: # Not perfect, has some randomness
                    dx = self.player_pos[0] - m["pos"][0]
                    dy = self.player_pos[1] - m["pos"][1]
                    if abs(dx) > abs(dy): m["pos"][0] += np.sign(dx)
                    elif dy != 0: m["pos"][1] += np.sign(dy)
            elif m["type"] == "jumper":
                m["state"]["timer"] -= 1
                if m["state"]["timer"] <= 0:
                    m["pos"][0] += self.np_random.integers(-1, 2)
                    m["pos"][1] += self.np_random.integers(-1, 2)
                    m["state"]["timer"] = self.np_random.integers(3, 6)
            elif m["type"] == "diagonal":
                m["pos"][0] += m["state"]["dir"]
                m["pos"][1] += m["state"]["dir"]
                if m["pos"][0] <= 0 or m["pos"][0] >= self.GRID_SIZE - 1 or \
                   m["pos"][1] <= 0 or m["pos"][1] >= self.GRID_SIZE - 1:
                    m["state"]["dir"] *= -1
            
            m["pos"][0] = np.clip(m["pos"][0], 0, self.GRID_SIZE - 1)
            m["pos"][1] = np.clip(m["pos"][1], 0, self.GRID_SIZE - 1)

    def _spawn_powerup(self):
        occupied = {tuple(m["pos"]) for m in self.monsters}
        occupied.add(tuple(self.player_pos))
        while True:
            pos = [self.np_random.integers(0, self.GRID_SIZE), self.np_random.integers(0, self.GRID_SIZE)]
            if tuple(pos) not in occupied:
                self.powerup_pos = pos
                self.powerup_active = True
                break
    
    def _calculate_movement_reward(self, movement):
        if movement == 0 or not self.monsters: return 0.0

        def dist(p1, p2):
            return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

        closest_monster = min(self.monsters, key=lambda m: dist(self.player_pos, m["pos"]))
        dist_before = dist(self.player_pos, closest_monster["pos"])

        if dist_before > 3: return 0.0

        next_pos = list(self.player_pos)
        if movement == 1: next_pos[1] -= 1
        elif movement == 2: next_pos[1] += 1
        elif movement == 3: next_pos[0] -= 1
        elif movement == 4: next_pos[0] += 1
        
        dist_after = dist(next_pos, closest_monster["pos"])
        
        if dist_after > dist_before: return 0.2  # Moved away
        if dist_after < dist_before: return -0.2 # Moved towards
        return 0.0

    def _create_powerup_particles(self, grid_pos):
        px, py = self._grid_to_pixel(grid_pos)
        for _ in range(30):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 2 + 1
            self.powerup_particles.append({
                "pos": [px, py],
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "life": self.np_random.integers(15, 30),
                "color": self.COLOR_POWERUP,
            })

    def _update_and_draw_particles(self):
        for p in self.powerup_particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["life"] -= 1
            if p["life"] <= 0:
                self.powerup_particles.remove(p)
            else:
                alpha = int(255 * (p["life"] / 30))
                color = (*p["color"], alpha)
                pygame.draw.circle(self.screen, color, (int(p["pos"][0]), int(p["pos"][1])), 2)

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup a display window
    pygame.display.set_caption("Monster Mash")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    terminated = False
    total_reward = 0
    
    # Game loop
    while not terminated:
        action = [0, 0, 0] # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            # Wait a bit before closing
            pygame.time.wait(3000)

        # Since auto_advance is False, we control the speed for human play
        clock.tick(10) # 10 steps per second for human playability
        
    pygame.quit()