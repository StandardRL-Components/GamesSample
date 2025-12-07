import gymnasium as gym
import os
import pygame
import numpy as np
from gymnasium.spaces import MultiDiscrete, Box
import pygame.gfxdraw
import math
import random
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    game_description = (
        "Match colorful tiles to power up your symbiotic microbes and defend against waves of incoming invaders."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the cursor. Press space to swap with the tile above, "
        "or shift to swap with the tile to the right. Match 3+ tiles to attack invaders."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Critical Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame and Display Setup ---
        self.WIDTH, self.HEIGHT = 640, 400
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.render_mode = render_mode

        # --- Game Constants ---
        self.BOARD_COLS, self.BOARD_ROWS = 10, 8
        self.TILE_SIZE = 40
        self.BOARD_WIDTH_PX = self.BOARD_COLS * self.TILE_SIZE
        self.BOARD_HEIGHT_PX = self.BOARD_ROWS * self.TILE_SIZE
        self.BOARD_ORIGIN_X = (self.WIDTH - self.BOARD_WIDTH_PX) // 2
        self.BOARD_ORIGIN_Y = 20
        self.MICROBE_ZONE_Y = self.BOARD_ORIGIN_Y + self.BOARD_HEIGHT_PX + 10
        self.NUM_TILE_TYPES = 5
        self.MAX_STEPS = 1000
        self.MICROBE_MAX_HEALTH = 100
        self.MICROBE_ATTACK_COOLDOWN = 15 # frames
        self.MICROBE_ATTACK_DAMAGE = 2.5

        # --- Visuals ---
        self.COLORS = {
            "BG": (10, 20, 35),
            "GRID": (30, 40, 55),
            "CURSOR": (255, 255, 255),
            "UI_TEXT": (200, 220, 255),
            "INVADER": (255, 50, 100),
            "INVADER_GLOW": (255, 100, 150),
            "PROJECTILE": (150, 255, 255),
        }
        self.TILE_COLORS = [
            (50, 200, 50),   # Green
            (50, 150, 255),  # Blue
            (255, 180, 50),  # Orange
            (200, 80, 255),  # Purple
            (255, 255, 80),  # Yellow
        ]
        self.font_main = pygame.font.SysFont('Consolas', 20, bold=True)
        self.font_large = pygame.font.SysFont('Consolas', 48, bold=True)
        
        # --- State Variables (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.board = None
        self.cursor_pos = None
        self.microbes = []
        self.invaders = []
        self.particles = []
        self.projectiles = []
        self.invader_spawn_timer = 0
        self.invader_spawn_rate = 60 # frames between spawns initially
        self.invader_base_health = 5

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        
        self.invader_spawn_rate = 60
        self.invader_spawn_timer = self.invader_spawn_rate
        self.invader_base_health = 5
        
        self._initialize_board()
        self._initialize_microbes()
        
        self.invaders = []
        self.particles = []
        self.projectiles = []
        
        self.cursor_pos = [self.BOARD_COLS // 2, self.BOARD_ROWS // 2]
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = -0.01  # Small penalty for existing

        # 1. Handle Player Input
        self._handle_input(action)
        
        # 2. Update Game Logic (Matches, Gravity)
        match_reward, match_score = self._update_board()
        reward += match_reward
        self.score += match_score
        
        # 3. Update Entities (Microbes, Invaders)
        kill_reward, kill_score = self._update_microbes_and_projectiles()
        reward += kill_reward
        self.score += kill_score

        damage_reward, damage_score = self._update_invaders()
        reward += damage_reward
        self.score += damage_score

        self._update_particles()
        
        # 4. Spawn New Entities
        self._spawn_invaders()
        
        # 5. Update Difficulty
        self._update_difficulty()
        
        # 6. Check for Termination
        terminated = False
        truncated = False
        if not any(m['health'] > 0 for m in self.microbes):
            self.game_over = True
            self.win = False
            terminated = True
            reward -= 100
        elif self.steps >= self.MAX_STEPS:
            self.game_over = True
            self.win = True
            truncated = True # Time limit reached
            reward += 100
            self.score += 5000

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    # --- Core Game Logic Sub-functions ---

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        if movement > 0:
            dx, dy = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)][movement]
            self.cursor_pos[0] = (self.cursor_pos[0] + dx) % self.BOARD_COLS
            self.cursor_pos[1] = (self.cursor_pos[1] + dy) % self.BOARD_ROWS
        
        if space_held: # Swap Up
            self._attempt_swap(self.cursor_pos, [self.cursor_pos[0], self.cursor_pos[1] - 1])
        if shift_held: # Swap Right
            self._attempt_swap(self.cursor_pos, [self.cursor_pos[0] + 1, self.cursor_pos[1]])

    def _attempt_swap(self, pos1, pos2):
        c1, r1 = pos1
        c2, r2 = pos2
        
        if not (0 <= c2 < self.BOARD_COLS and 0 <= r2 < self.BOARD_ROWS):
            return False

        self.board[r1, c1], self.board[r2, c2] = self.board[r2, c2], self.board[r1, c1]
        
        if self._has_match_at([pos1, pos2]):
            return True
        else:
            self.board[r1, c1], self.board[r2, c2] = self.board[r2, c2], self.board[r1, c1]
            return False

    def _update_board(self):
        total_matched_tiles = 0
        reward = 0
        
        while True:
            matches = self._find_all_matches()
            if not matches:
                break

            num_matched = len(matches)
            total_matched_tiles += num_matched
            reward += 0.1 * num_matched
            
            matched_types = set()
            for r, c in matches:
                if self.board[r, c] != -1:
                    matched_types.add(self.board[r, c])
                    self._create_particles(c, r, self.TILE_COLORS[self.board[r, c]], 15, 2.5)
                    self.board[r, c] = -1

            for tile_type in matched_types:
                for microbe in self.microbes:
                    if microbe['type'] == tile_type:
                        microbe['powered_timer'] = 90 # 3 seconds at 30fps
            
            self._apply_gravity()
            self._refill_board()
        
        return reward, total_matched_tiles * 10

    def _update_microbes_and_projectiles(self):
        reward = 0
        score = 0
        
        # Update and fire microbes
        for i, microbe in enumerate(self.microbes):
            microbe['powered_timer'] = max(0, microbe['powered_timer'] - 1)
            microbe['attack_cooldown'] = max(0, microbe['attack_cooldown'] - 1)
            
            if microbe['health'] > 0 and microbe['powered_timer'] > 0 and microbe['attack_cooldown'] == 0:
                target = None
                min_dist = float('inf')
                for invader in self.invaders:
                    dist = abs(invader['pos'][0] - (self.BOARD_ORIGIN_X + i * self.TILE_SIZE + self.TILE_SIZE // 2))
                    if dist < min_dist:
                        min_dist = dist
                        target = invader
                
                if target:
                    start_x = self.BOARD_ORIGIN_X + i * self.TILE_SIZE + self.TILE_SIZE // 2
                    start_y = self.MICROBE_ZONE_Y
                    self.projectiles.append({'pos': [start_x, start_y], 'target': target, 'type': microbe['type']})
                    microbe['attack_cooldown'] = self.MICROBE_ATTACK_COOLDOWN

        # Update projectiles
        for proj in self.projectiles[:]:
            target_pos = proj['target']['pos']
            proj_pos = proj['pos']
            
            dist = math.hypot(target_pos[0] - proj_pos[0], target_pos[1] - proj_pos[1])
            if dist < 10:
                proj['target']['health'] -= self.MICROBE_ATTACK_DAMAGE
                self._create_particles(proj['target']['pos'][0], proj['target']['pos'][1], self.TILE_COLORS[proj['type']], 10, 1.5)
                self.projectiles.remove(proj)
                if proj['target']['health'] <= 0 and proj['target'] in self.invaders:
                    self._create_particles(proj['target']['pos'][0], proj['target']['pos'][1], self.COLORS['INVADER_GLOW'], 30, 3.0)
                    self.invaders.remove(proj['target'])
                    reward += 1.0
                    score += 50
            else:
                angle = math.atan2(target_pos[1] - proj_pos[1], target_pos[0] - proj_pos[0])
                proj_pos[0] += math.cos(angle) * 8
                proj_pos[1] += math.sin(angle) * 8
                
        return reward, score

    def _update_invaders(self):
        reward = 0
        score = 0
        for invader in self.invaders[:]:
            invader['pos'][1] += 0.5 # Invader speed
            
            if invader['pos'][1] > self.MICROBE_ZONE_Y - 10:
                col = int((invader['pos'][0] - self.BOARD_ORIGIN_X) / self.TILE_SIZE)
                if 0 <= col < self.BOARD_COLS:
                    microbe = self.microbes[col]
                    if microbe['health'] > 0:
                        damage = 25
                        microbe['health'] = max(0, microbe['health'] - damage)
                        reward -= 1.0
                        score -= 25
                        self._create_particles(invader['pos'][0], invader['pos'][1], self.COLORS['INVADER'], 20, 2.0)
                        self.invaders.remove(invader)

        return reward, score

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _spawn_invaders(self):
        self.invader_spawn_timer -= 1
        if self.invader_spawn_timer <= 0:
            x = self.BOARD_ORIGIN_X + self.np_random.random() * self.BOARD_WIDTH_PX
            y = self.BOARD_ORIGIN_Y - 20
            health = self.invader_base_health
            self.invaders.append({'pos': [x, y], 'health': health, 'max_health': health})
            self.invader_spawn_timer = self.invader_spawn_rate

    def _update_difficulty(self):
        if self.steps > 0 and self.steps % 200 == 0:
            self.invader_spawn_rate = max(15, self.invader_spawn_rate * 0.95)
        if self.steps > 0 and self.steps % 500 == 0:
            self.invader_base_health += 1

    # --- Helper and Initialization functions ---
    
    def _initialize_board(self):
        while True:
            self.board = self.np_random.integers(0, self.NUM_TILE_TYPES, size=(self.BOARD_ROWS, self.BOARD_COLS))
            if not self._find_all_matches():
                break
    
    def _initialize_microbes(self):
        self.microbes = []
        for i in range(self.BOARD_COLS):
            self.microbes.append({
                "type": self.np_random.integers(0, self.NUM_TILE_TYPES),
                "health": self.MICROBE_MAX_HEALTH,
                "max_health": self.MICROBE_MAX_HEALTH,
                "powered_timer": 0,
                "attack_cooldown": 0,
            })
            
    def _has_match_at(self, positions):
        for r, c in positions:
            if not (0 <= r < self.BOARD_ROWS and 0 <= c < self.BOARD_COLS): continue
            tile_type = self.board[r, c]
            if tile_type == -1: continue
            # Horizontal check
            count = 1
            for i in range(1, 3):
                if c - i >= 0 and self.board[r, c - i] == tile_type: count += 1
                else: break
            for i in range(1, 3):
                if c + i < self.BOARD_COLS and self.board[r, c + i] == tile_type: count += 1
                else: break
            if count >= 3: return True
            
            # Vertical check
            count = 1
            for i in range(1, 3):
                if r - i >= 0 and self.board[r - i, c] == tile_type: count += 1
                else: break
            for i in range(1, 3):
                if r + i < self.BOARD_ROWS and self.board[r + i, c] == tile_type: count += 1
                else: break
            if count >= 3: return True
        return False
        
    def _find_all_matches(self):
        matches = set()
        for r in range(self.BOARD_ROWS):
            for c in range(self.BOARD_COLS - 2):
                if self.board[r,c] != -1 and self.board[r,c] == self.board[r,c+1] == self.board[r,c+2]:
                    for i in range(3): matches.add((r, c+i))
        for c in range(self.BOARD_COLS):
            for r in range(self.BOARD_ROWS - 2):
                if self.board[r,c] != -1 and self.board[r,c] == self.board[r+1,c] == self.board[r+2,c]:
                    for i in range(3): matches.add((r+i, c))
        return list(matches)

    def _apply_gravity(self):
        for c in range(self.BOARD_COLS):
            write_idx = self.BOARD_ROWS - 1
            for r in range(self.BOARD_ROWS - 1, -1, -1):
                if self.board[r, c] != -1:
                    if r != write_idx:
                        self.board[write_idx, c] = self.board[r, c]
                        self.board[r, c] = -1
                    write_idx -= 1
                    
    def _refill_board(self):
        for r in range(self.BOARD_ROWS):
            for c in range(self.BOARD_COLS):
                if self.board[r, c] == -1:
                    self.board[r, c] = self.np_random.integers(0, self.NUM_TILE_TYPES)
                    
    def _create_particles(self, x, y, color, count, speed_mult=1.0):
        is_board_particle = 0 <= x < self.BOARD_COLS and 0 <= y < self.BOARD_ROWS
        px, py = (self.BOARD_ORIGIN_X + x * self.TILE_SIZE + self.TILE_SIZE//2, self.BOARD_ORIGIN_Y + y * self.TILE_SIZE + self.TILE_SIZE//2) if is_board_particle else (x, y)
        
        for _ in range(count):
            angle = self.np_random.random() * 2 * math.pi
            speed = (self.np_random.random() * 2 + 1) * speed_mult
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({'pos': [px, py], 'vel': vel, 'life': self.np_random.integers(15, 30), 'color': color})

    # --- Rendering ---

    def _get_observation(self):
        self.screen.fill(self.COLORS["BG"])
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "invaders": len(self.invaders)}

    def render(self):
        return self._get_observation()

    def _render_game(self):
        self._draw_grid()
        self._draw_tiles()
        self._draw_cursor()
        self._draw_microbes()
        self._draw_projectiles()
        self._draw_invaders()
        self._draw_particles()

    def _draw_grid(self):
        for r in range(self.BOARD_ROWS + 1):
            y = self.BOARD_ORIGIN_Y + r * self.TILE_SIZE
            pygame.draw.line(self.screen, self.COLORS["GRID"], (self.BOARD_ORIGIN_X, y), (self.BOARD_ORIGIN_X + self.BOARD_WIDTH_PX, y))
        for c in range(self.BOARD_COLS + 1):
            x = self.BOARD_ORIGIN_X + c * self.TILE_SIZE
            pygame.draw.line(self.screen, self.COLORS["GRID"], (x, self.BOARD_ORIGIN_Y), (x, self.BOARD_ORIGIN_Y + self.BOARD_HEIGHT_PX))
            
    def _draw_tiles(self):
        for r in range(self.BOARD_ROWS):
            for c in range(self.BOARD_COLS):
                tile_type = self.board[r, c]
                if tile_type != -1:
                    rect = pygame.Rect(self.BOARD_ORIGIN_X + c * self.TILE_SIZE + 2, self.BOARD_ORIGIN_Y + r * self.TILE_SIZE + 2, self.TILE_SIZE - 4, self.TILE_SIZE - 4)
                    color = self.TILE_COLORS[tile_type]
                    pygame.gfxdraw.box(self.screen, rect, color)
                    
    def _draw_cursor(self):
        c, r = self.cursor_pos
        pulse = (math.sin(pygame.time.get_ticks() * 0.01) + 1) / 2 * 5
        rect = pygame.Rect(self.BOARD_ORIGIN_X + c * self.TILE_SIZE, self.BOARD_ORIGIN_Y + r * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
        pygame.draw.rect(self.screen, self.COLORS["CURSOR"], rect, int(3 + pulse))
        
    def _draw_microbes(self):
        for i, microbe in enumerate(self.microbes):
            x = self.BOARD_ORIGIN_X + i * self.TILE_SIZE + self.TILE_SIZE // 2
            y = self.MICROBE_ZONE_Y + 15
            color = self.TILE_COLORS[microbe['type']]
            
            if microbe['health'] > 0:
                pygame.gfxdraw.aacircle(self.screen, int(x), int(y), 12, color)
                pygame.gfxdraw.filled_circle(self.screen, int(x), int(y), 12, color)
                if microbe['powered_timer'] > 0:
                    alpha = 100 + (microbe['powered_timer'] / 90) * 155
                    glow_color = (*color, int(alpha / 4))
                    pygame.gfxdraw.filled_circle(self.screen, int(x), int(y), 18, glow_color)
                
                # Health bar
                health_pct = microbe['health'] / microbe['max_health']
                bar_width = self.TILE_SIZE - 10
                bar_x = self.BOARD_ORIGIN_X + i * self.TILE_SIZE + 5
                bar_y = y + 20
                pygame.draw.rect(self.screen, (255,0,0), (bar_x, bar_y, bar_width, 5))
                pygame.draw.rect(self.screen, (0,255,0), (bar_x, bar_y, int(bar_width * health_pct), 5))

    def _draw_invaders(self):
        for invader in self.invaders:
            x, y = int(invader['pos'][0]), int(invader['pos'][1])
            size = 10
            points = [(x, y - size), (x + size//2, y), (x, y + size), (x - size//2, y)]
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLORS['INVADER_GLOW'])
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLORS['INVADER'])

    def _draw_projectiles(self):
        for proj in self.projectiles:
            x, y = int(proj['pos'][0]), int(proj['pos'][1])
            color = self.COLORS['PROJECTILE']
            pygame.gfxdraw.aacircle(self.screen, x, y, 4, color)
            pygame.gfxdraw.filled_circle(self.screen, x, y, 4, color)
            
    def _draw_particles(self):
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / 30.0))))
            color = (*p['color'], alpha)
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            radius = int(p['life'] / 6)
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, color)

    def _render_ui(self):
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLORS["UI_TEXT"])
        self.screen.blit(score_text, (10, 10))
        
        steps_text = self.font_main.render(f"TIME: {self.steps}/{self.MAX_STEPS}", True, self.COLORS["UI_TEXT"])
        self.screen.blit(steps_text, (self.WIDTH - steps_text.get_width() - 10, 10))

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0,0,0,180))
            self.screen.blit(overlay, (0,0))
            msg = "YOU WIN!" if self.win else "GAME OVER"
            end_text = self.font_large.render(msg, True, (255,255,255))
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

if __name__ == '__main__':
    # This block is for human play and visualization
    # It is not part of the required Gymnasium interface
    # but is useful for testing and debugging.
    # To use, you might need to `pip install pygame`
    # and unset the dummy video driver.
    if os.environ.get("SDL_VIDEODRIVER") == "dummy":
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array") 
    pygame.display.set_caption("Symbiotic Defense")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    obs, info = env.reset()
    done = False
    
    running = True
    while running:
        action = [0, 0, 0] # Default no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: action[0] = 1
                elif event.key == pygame.K_DOWN: action[0] = 2
                elif event.key == pygame.K_LEFT: action[0] = 3
                elif event.key == pygame.K_RIGHT: action[0] = 4
                elif event.key == pygame.K_SPACE: action[1] = 1
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: action[2] = 1
                elif event.key == pygame.K_r: # Reset
                    obs, info = env.reset()
                    done = False
                    
        if not done:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        display_surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(display_surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30)

    pygame.quit()