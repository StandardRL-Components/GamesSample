import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:29:48.089931
# Source Brief: brief_01093.md
# Brief Index: 1093
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

# Helper classes defined outside the main environment class
class Fighter:
    def __init__(self, fighter_type, start_pos, target_x, owner):
        self.type = fighter_type
        self.x, self.y = start_pos
        self.target_x = target_x
        self.hp = self.type['hp']
        self.owner = owner
        self.speed = self.type['speed'] if owner == 'player' else -self.type['speed']
        self.color = self.type['color']
        self.size = 10

    def update(self):
        self.x += self.speed
        # Small random vertical drift to make movement less static
        self.y += random.uniform(-0.5, 0.5)

    def draw(self, screen):
        # Draw a triangle pointing in the direction of movement
        p1 = (int(self.x + self.size * (1 if self.owner == 'player' else -1)), int(self.y))
        p2 = (int(self.x - self.size/2 * (1 if self.owner == 'player' else -1)), int(self.y - self.size))
        p3 = (int(self.x - self.size/2 * (1 if self.owner == 'player' else -1)), int(self.y + self.size))
        
        # Glow effect
        glow_color = tuple(min(255, c + 50) for c in self.color)
        pygame.gfxdraw.filled_trigon(screen, p1[0], p1[1], p2[0], p2[1], p3[0], p3[1], glow_color)
        pygame.gfxdraw.aatrigon(screen, p1[0], p1[1], p2[0], p2[1], p3[0], p3[1], glow_color)
        
        # Main body
        pygame.gfxdraw.filled_trigon(screen, p1[0], p1[1], p2[0], p2[1], p3[0], p3[1], self.color)
        pygame.gfxdraw.aatrigon(screen, p1[0], p1[1], p2[0], p2[1], p3[0], p3[1], self.color)


class Particle:
    def __init__(self, x, y, color, life):
        self.x = x
        self.y = y
        self.vx = random.uniform(-1, 1)
        self.vy = random.uniform(-1, 1)
        self.life = life
        self.max_life = life
        self.color = color

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.life -= 1

    def draw(self, screen):
        if self.life > 0:
            alpha = int(255 * (self.life / self.max_life))
            size = int(5 * (self.life / self.max_life))
            if size > 0:
                s = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(s, (*self.color, alpha), (size, size), size)
                screen.blit(s, (int(self.x - size), int(self.y - size)))

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Match colorful tiles to generate energy, then deploy a variety of fighters to overwhelm your opponent's portal in this strategic puzzle-brawler."
    )
    user_guide = (
        "Controls: Use ←→↑↓ to move the cursor and swap tiles. Match 3 to earn energy. "
        "When deploying, use ↑↓ to aim, Shift to cycle units, and Space to deploy."
    )
    auto_advance = True

    # Class-level state for persistent unlocks
    FIGHTER_CATALOG = [
        {'name': 'Scout', 'cost': 20, 'hp': 10, 'damage': 5, 'speed': 2.0, 'color': (100, 255, 100)},
        {'name': 'Tank', 'cost': 40, 'hp': 30, 'damage': 10, 'speed': 1.0, 'color': (50, 200, 255)},
        {'name': 'Striker', 'cost': 60, 'hp': 15, 'damage': 20, 'speed': 1.5, 'color': (255, 200, 50)},
    ]
    unlocked_fighter_indices = {0}
    wins = 0

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Spaces ---
        self.observation_space = Box(low=0, high=255, shape=(400, 640, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 16)
        self.font_medium = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)

        # --- Colors ---
        self.COLOR_BG = (15, 20, 35)
        self.COLOR_GRID_BG = (25, 30, 45)
        self.COLOR_PLAYER = (50, 255, 150)
        self.COLOR_OPPONENT = (255, 50, 100)
        self.COLOR_ENERGY = (50, 150, 255)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.TILE_COLORS = [
            (255, 80, 80), (80, 255, 80), (80, 80, 255),
            (255, 255, 80), (255, 80, 255), (80, 255, 255)
        ]

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 5000
        self.BOARD_ROWS, self.BOARD_COLS = 8, 8
        self.TILE_SIZE = 28
        self.BOARD_X, self.BOARD_Y = 30, 80
        self.MAX_HEALTH = 100
        self.MAX_ENERGY = 100
        self.DEPLOY_AREA_Y_RANGE = (50, 350)
        self.PLAYER_PORTAL_X = 260
        self.OPPONENT_PORTAL_X = 380
        
        # --- State variables will be initialized in reset() ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_phase = 'MATCHING' # 'MATCHING' or 'DEPLOYING'
        self.player_health = 0
        self.opponent_health = 0
        self.player_energy = 0
        self.opponent_energy = 0
        self.board = None
        self.cursor_pos = [0, 0]
        self.selected_tile = None
        self.active_fighters = []
        self.particles = []
        self.deployment_timer = 0
        self.no_match_timer = 0
        self.opponent_ai_energy_rate = 0.2
        self.available_fighters = []
        self.selected_fighter_idx = 0
        self.aim_pos = [self.PLAYER_PORTAL_X, self.HEIGHT / 2]
        self.prev_space_held = False
        self.prev_shift_held = False
        self.last_actions = deque(maxlen=10)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_phase = 'MATCHING'
        
        self.player_health = self.MAX_HEALTH
        self.opponent_health = self.MAX_HEALTH
        self.player_energy = 0
        self.opponent_energy = 0

        self._init_board()
        self.cursor_pos = [self.BOARD_ROWS // 2, self.BOARD_COLS // 2]
        self.selected_tile = None
        
        self.active_fighters = []
        self.particles = []
        
        self.deployment_timer = 0
        self.no_match_timer = 0
        self.opponent_ai_energy_rate = 0.2
        
        self.available_fighters = [ft for i, ft in enumerate(self.FIGHTER_CATALOG) if i in self.unlocked_fighter_indices]
        self.selected_fighter_idx = 0
        self.aim_pos = [self.PLAYER_PORTAL_X, self.HEIGHT / 2]
        
        self.prev_space_held = False
        self.prev_shift_held = False
        self.last_actions.clear()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.steps += 1
        reward = 0

        # --- Handle Input ---
        reward += self._handle_actions(action)
        
        # --- Update Game State ---
        self._update_fighters()
        self._update_ai()
        self._update_timers()
        self._update_particles()
        
        # --- Calculate Step Reward ---
        # This is tricky because damage is event-based. We'll add it in the relevant methods.
        # Here we add a small time penalty to encourage faster play.
        reward -= 0.01

        # --- Check Termination ---
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        if terminated or truncated:
            if self.player_health <= 0:
                reward -= 100
                self.score -= 100
            elif self.opponent_health <= 0:
                reward += 100
                self.score += 100
                GameEnv.wins += 1
                if GameEnv.wins % 5 == 0:
                    # Unlock a new fighter
                    if len(self.unlocked_fighter_indices) < len(self.FIGHTER_CATALOG):
                        self.unlocked_fighter_indices.add(len(self.unlocked_fighter_indices))
            self.game_over = True
            
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_actions(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.prev_space_held
        shift_pressed = shift_held and not self.prev_shift_held
        self.prev_space_held, self.prev_shift_held = space_held, shift_held
        
        action_reward = 0

        if self.game_phase == 'MATCHING':
            # --- Move Cursor ---
            if movement == 1: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
            if movement == 2: self.cursor_pos[0] = min(self.BOARD_ROWS - 1, self.cursor_pos[0] + 1)
            if movement == 3: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
            if movement == 4: self.cursor_pos[1] = min(self.BOARD_COLS - 1, self.cursor_pos[1] + 1)
            
            # --- Select/Swap Tiles ---
            if space_pressed:
                if self.selected_tile is None:
                    self.selected_tile = tuple(self.cursor_pos)
                else:
                    # Check for valid adjacent swap
                    dist = abs(self.selected_tile[0] - self.cursor_pos[0]) + abs(self.selected_tile[1] - self.cursor_pos[1])
                    if dist == 1:
                        # Swap
                        r1, c1 = self.selected_tile
                        r2, c2 = self.cursor_pos
                        self.board[r1, c1], self.board[r2, c2] = self.board[r2, c2], self.board[r1, c1]
                        
                        # Check for matches
                        matches = self._find_and_handle_matches()
                        if matches:
                            action_reward += len(matches) * 0.1 # Reward for each matched tile
                            self.player_energy = min(self.MAX_ENERGY, self.player_energy + len(matches) * 2)
                            self.no_match_timer = 0
                            # sound placeholder: # sfx_match.play()
                        else:
                            # Invalid swap, swap back
                            self.board[r1, c1], self.board[r2, c2] = self.board[r2, c2], self.board[r1, c1]
                            # sound placeholder: # sfx_invalid_swap.play()
                    self.selected_tile = None
        
        elif self.game_phase == 'DEPLOYING':
            # --- Move Aiming Reticle ---
            if movement == 1: self.aim_pos[1] = max(self.DEPLOY_AREA_Y_RANGE[0], self.aim_pos[1] - 5)
            if movement == 2: self.aim_pos[1] = min(self.DEPLOY_AREA_Y_RANGE[1], self.aim_pos[1] + 5)
            
            # --- Cycle Fighters ---
            if shift_pressed and len(self.available_fighters) > 1:
                self.selected_fighter_idx = (self.selected_fighter_idx + 1) % len(self.available_fighters)
                # sound placeholder: # sfx_cycle_fighter.play()

            # --- Deploy Fighter ---
            if space_pressed:
                fighter_type = self.available_fighters[self.selected_fighter_idx]
                if self.player_energy >= fighter_type['cost']:
                    self.player_energy -= fighter_type['cost']
                    new_fighter = Fighter(fighter_type, (self.PLAYER_PORTAL_X, self.aim_pos[1]), self.OPPONENT_PORTAL_X, 'player')
                    self.active_fighters.append(new_fighter)
                    self._create_particles(self.aim_pos[0], self.aim_pos[1], fighter_type['color'], 20)
                    # sound placeholder: # sfx_deploy.play()
                    self.game_phase = 'MATCHING' # Switch back after deploying
        
        return action_reward
        
    def _update_fighters(self):
        fighters_to_remove = []
        reward = 0
        for fighter in self.active_fighters:
            fighter.update()
            if fighter.owner == 'player' and fighter.x > self.OPPONENT_PORTAL_X:
                self.opponent_health = max(0, self.opponent_health - fighter.type['damage'])
                reward += 5 # Reward for damaging opponent
                self.score += 5
                self._create_particles(fighter.x, fighter.y, self.COLOR_OPPONENT, 30)
                fighters_to_remove.append(fighter)
                # sound placeholder: # sfx_opponent_damage.play()
            elif fighter.owner == 'opponent' and fighter.x < self.PLAYER_PORTAL_X:
                self.player_health = max(0, self.player_health - fighter.type['damage'])
                reward -= 1 # Penalty for taking damage
                self.score -= 1
                self._create_particles(fighter.x, fighter.y, self.COLOR_PLAYER, 30)
                fighters_to_remove.append(fighter)
                # sound placeholder: # sfx_player_damage.play()
        
        self.active_fighters = [f for f in self.active_fighters if f not in fighters_to_remove]
        return reward

    def _update_ai(self):
        # AI gains energy over time, rate increases with game steps
        scaled_rate = self.opponent_ai_energy_rate + (self.steps // 500) * 0.05
        self.opponent_energy = min(self.MAX_ENERGY, self.opponent_energy + scaled_rate)
        
        # AI deployment logic
        deployable_fighters = [f for f in self.FIGHTER_CATALOG if self.opponent_energy >= f['cost']]
        if deployable_fighters:
            # Prioritize highest damage fighter it can afford
            fighter_to_deploy = max(deployable_fighters, key=lambda f: f['damage'])
            self.opponent_energy -= fighter_to_deploy['cost']
            
            spawn_y = random.uniform(*self.DEPLOY_AREA_Y_RANGE)
            new_fighter = Fighter(fighter_to_deploy, (self.OPPONENT_PORTAL_X, spawn_y), self.PLAYER_PORTAL_X, 'opponent')
            self.active_fighters.append(new_fighter)
            self._create_particles(self.OPPONENT_PORTAL_X, spawn_y, fighter_to_deploy['color'], 20)
            # sound placeholder: # sfx_opponent_deploy.play()
            return -0.1 # Small penalty for opponent action
        return 0

    def _update_timers(self):
        # Deployment phase timer
        can_afford_fighter = any(self.player_energy >= f['cost'] for f in self.available_fighters)
        if self.game_phase == 'MATCHING' and can_afford_fighter:
            self.game_phase = 'DEPLOYING'
            self.deployment_timer = 5 * 30 # 5 seconds at 30 FPS
        
        if self.game_phase == 'DEPLOYING':
            self.deployment_timer -= 1
            if self.deployment_timer <= 0:
                self.game_phase = 'MATCHING'

        # No-match shuffle timer
        self.no_match_timer += 1
        if self.no_match_timer > 10 * 30: # 10 seconds
            self._init_board() # Shuffle by re-initializing
            self.no_match_timer = 0

    def _update_particles(self):
        self.particles = [p for p in self.particles if p.life > 0]
        for p in self.particles:
            p.update()

    def _check_termination(self):
        return self.player_health <= 0 or self.opponent_health <= 0

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    # --- Rendering Methods ---
    def _render_game(self):
        # Draw battlefield dividers
        pygame.draw.line(self.screen, self.COLOR_GRID_BG, (self.PLAYER_PORTAL_X, 0), (self.PLAYER_PORTAL_X, self.HEIGHT), 3)
        pygame.draw.line(self.screen, self.COLOR_GRID_BG, (self.OPPONENT_PORTAL_X, 0), (self.OPPONENT_PORTAL_X, self.HEIGHT), 3)

        self._render_board()
        for fighter in self.active_fighters:
            fighter.draw(self.screen)
        for particle in self.particles:
            particle.draw(self.screen)
        
        if self.game_phase == 'DEPLOYING':
            self._render_aim_reticle()

    def _render_board(self):
        board_width = self.BOARD_COLS * self.TILE_SIZE
        board_height = self.BOARD_ROWS * self.TILE_SIZE
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, (self.BOARD_X, self.BOARD_Y, board_width, board_height))

        for r in range(self.BOARD_ROWS):
            for c in range(self.BOARD_COLS):
                tile_color = self.TILE_COLORS[self.board[r, c]]
                rect = (self.BOARD_X + c * self.TILE_SIZE, self.BOARD_Y + r * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
                pygame.draw.rect(self.screen, tile_color, rect, border_radius=4)
                pygame.draw.rect(self.screen, tuple(min(255, x+30) for x in tile_color), rect, width=2, border_radius=4)

        # Draw cursor
        cursor_r, cursor_c = self.cursor_pos
        cursor_rect = (self.BOARD_X + cursor_c * self.TILE_SIZE, self.BOARD_Y + cursor_r * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
        pygame.draw.rect(self.screen, (255, 255, 255), cursor_rect, 3, border_radius=4)
        
        # Draw selected tile highlight
        if self.selected_tile is not None:
            r, c = self.selected_tile
            selected_rect = (self.BOARD_X + c * self.TILE_SIZE, self.BOARD_Y + r * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
            pygame.draw.rect(self.screen, (255, 255, 0), selected_rect, 3, border_radius=4)

    def _render_aim_reticle(self):
        x, y = self.aim_pos
        color = self.COLOR_PLAYER
        pygame.draw.circle(self.screen, color, (int(x), int(y)), 15, 2)
        pygame.draw.line(self.screen, color, (int(x) - 20, int(y)), (int(x) + 20, int(y)), 2)
        pygame.draw.line(self.screen, color, (int(x), int(y) - 20), (int(x), int(y) + 20), 2)

    def _render_ui(self):
        # Health Bars
        self._render_bar(10, 10, 240, 20, self.player_health / self.MAX_HEALTH, self.COLOR_PLAYER, f"PLAYER: {self.player_health}/{self.MAX_HEALTH}")
        self._render_bar(self.WIDTH - 250, 10, 240, 20, self.opponent_health / self.MAX_HEALTH, self.COLOR_OPPONENT, f"OPPONENT: {self.opponent_health}/{self.MAX_HEALTH}")
        
        # Energy Bars
        self._render_bar(self.BOARD_X, self.BOARD_Y + self.BOARD_ROWS * self.TILE_SIZE + 10, self.BOARD_COLS * self.TILE_SIZE, 15, self.player_energy / self.MAX_ENERGY, self.COLOR_ENERGY, f"ENERGY: {int(self.player_energy)}")
        opp_board_x = self.WIDTH - self.BOARD_X - self.BOARD_COLS * self.TILE_SIZE
        opp_board_y = self.BOARD_Y
        self._render_bar(opp_board_x, opp_board_y + self.BOARD_ROWS * self.TILE_SIZE + 10, self.BOARD_COLS * self.TILE_SIZE, 15, self.opponent_energy / self.MAX_ENERGY, self.COLOR_ENERGY, f"ENERGY: {int(self.opponent_energy)}")

        # Deployment UI
        if self.game_phase == 'DEPLOYING':
            fighter = self.available_fighters[self.selected_fighter_idx]
            text = f"Deploy: {fighter['name']} (Cost: {fighter['cost']})"
            self._draw_text(text, self.font_medium, self.COLOR_UI_TEXT, self.WIDTH / 2, self.HEIGHT - 50, center=True)
            timer_text = f"Time: {self.deployment_timer / 30:.1f}s"
            self._draw_text(timer_text, self.font_small, self.COLOR_UI_TEXT, self.WIDTH / 2, self.HEIGHT - 25, center=True)

        if self.game_over:
            outcome = "VICTORY" if self.player_health > 0 else "DEFEAT"
            color = self.COLOR_PLAYER if self.player_health > 0 else self.COLOR_OPPONENT
            self._draw_text(outcome, self.font_large, color, self.WIDTH / 2, self.HEIGHT / 2, center=True)

    def _render_bar(self, x, y, w, h, progress, color, text):
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, (x, y, w, h), border_radius=4)
        pygame.draw.rect(self.screen, color, (x, y, w * max(0, progress), h), border_radius=4)
        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, (x, y, w, h), 2, border_radius=4)
        self._draw_text(text, self.font_small, self.COLOR_UI_TEXT, x + w / 2, y + h / 2, center=True)

    def _draw_text(self, text, font, color, x, y, center=False):
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        if center:
            text_rect.center = (int(x), int(y))
        else:
            text_rect.topleft = (int(x), int(y))
        self.screen.blit(text_surface, text_rect)

    # --- Tile Logic Helpers ---
    def _init_board(self):
        self.board = self.np_random.integers(0, len(self.TILE_COLORS), size=(self.BOARD_ROWS, self.BOARD_COLS))
        while self._find_matches():
            self._find_and_handle_matches() # Clear initial matches
        self.no_match_timer = 0
    
    def _find_matches(self):
        matches = set()
        for r in range(self.BOARD_ROWS):
            for c in range(self.BOARD_COLS):
                color = self.board[r, c]
                # Horizontal
                if c < self.BOARD_COLS - 2 and self.board[r, c+1] == color and self.board[r, c+2] == color:
                    matches.add((r, c)); matches.add((r, c+1)); matches.add((r, c+2))
                # Vertical
                if r < self.BOARD_ROWS - 2 and self.board[r+1, c] == color and self.board[r+2, c] == color:
                    matches.add((r, c)); matches.add((r+1, c)); matches.add((r+2, c))
        return list(matches)

    def _find_and_handle_matches(self):
        all_matches = set()
        while True:
            matches = self._find_matches()
            if not matches:
                break
            all_matches.update(matches)
            for r, c in matches:
                self._create_particles(self.BOARD_X + c * self.TILE_SIZE + self.TILE_SIZE/2,
                                       self.BOARD_Y + r * self.TILE_SIZE + self.TILE_SIZE/2,
                                       self.TILE_COLORS[self.board[r,c]], 5)
                self.board[r, c] = -1 # Mark for removal
            
            # Gravity
            for c in range(self.BOARD_COLS):
                empty_row = self.BOARD_ROWS - 1
                for r in range(self.BOARD_ROWS - 1, -1, -1):
                    if self.board[r, c] != -1:
                        self.board[empty_row, c], self.board[r, c] = self.board[r, c], self.board[empty_row, c]
                        empty_row -= 1
            
            # Refill
            for r in range(self.BOARD_ROWS):
                for c in range(self.BOARD_COLS):
                    if self.board[r, c] == -1:
                        self.board[r, c] = self.np_random.integers(0, len(self.TILE_COLORS))
        return list(all_matches)

    def _create_particles(self, x, y, color, count):
        for _ in range(count):
            self.particles.append(Particle(x, y, color, life=random.randint(15, 30)))

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # --- Manual Play Script ---
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Need a separate pygame window for display
    pygame.display.init()
    display_screen = pygame.display.set_mode((640, 400))
    pygame.display.set_caption("Portal Puzzle Fighters")

    # Game loop
    running = True
    while running:
        # --- Pygame event handling for manual control ---
        movement, space, shift = 0, 0, 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
        
        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Steps: {info['steps']}")
            # Render one last time to show the "VICTORY/DEFEAT" message
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            display_screen.blit(surf, (0, 0))
            pygame.display.flip()
            pygame.time.wait(2000) # Pause for 2 seconds
            
            obs, info = env.reset()

        # --- Display the environment's rendering ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Limit frame rate
        env.clock.tick(30)

    env.close()