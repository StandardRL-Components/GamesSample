import gymnasium as gym
import os
import pygame
import pygame.gfxdraw
import math
import random
from gymnasium.spaces import MultiDiscrete
import numpy as np
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Helper function for linear interpolation
def lerp(a, b, t):
    return a + (b - a) * t

# Helper function to draw text
def draw_text(surface, text, pos, font, color, center=True):
    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect()
    if center:
        text_rect.center = pos
    else:
        text_rect.topleft = pos
    surface.blit(text_surface, text_rect)

class QuantumCard:
    def __init__(self, row, col, grid_size, screen_dims, np_random):
        self.row, self.col = row, col
        self.grid_w, self.grid_h = grid_size
        self.screen_w, self.screen_h = screen_dims
        self.np_random = np_random
        
        # Superposition state
        self.alpha = 1 / math.sqrt(2)
        self.beta = 1 / math.sqrt(2)
        self.is_superposition = True
        self.collapsed_state = None # 0 or 1

        # Visuals
        self.size = 60
        self.padding = 20
        grid_width = self.grid_w * (self.size + self.padding) - self.padding
        grid_height = self.grid_h * (self.size + self.padding) - self.padding
        start_x = (self.screen_w - grid_width) / 2
        start_y = (self.screen_h - grid_height) / 2 + 50

        self.target_pos = pygame.Vector2(
            start_x + col * (self.size + self.padding),
            start_y + row * (self.size + self.padding)
        )
        self.render_pos = pygame.Vector2(self.target_pos)
        self.color = pygame.Color(0, 0, 0)
        self.anim_state = 'idle'
        self.anim_progress = 0.0

    def collapse(self, state):
        if self.is_superposition:
            self.is_superposition = False
            self.collapsed_state = state
            self.alpha = 1.0 if state == 0 else 0.0
            self.beta = 1.0 if state == 1 else 0.0
            self.anim_state = 'collapse'
            self.anim_progress = 0.0

    def update(self, dt):
        self.render_pos.x = lerp(self.render_pos.x, self.target_pos.x, dt * 10)
        self.render_pos.y = lerp(self.render_pos.y, self.target_pos.y, dt * 10)
        
        p_zero = self.alpha**2
        self.color.r = int(lerp(100, 255, 1 - p_zero))
        self.color.g = 50
        self.color.b = int(lerp(100, 255, p_zero))

        if self.anim_state != 'idle':
            self.anim_progress += dt * 2.0
            if self.anim_progress >= 1.0:
                self.anim_state = 'idle'
                self.anim_progress = 0.0


class Particle:
    def __init__(self, x, y, color):
        self.pos = pygame.Vector2(x, y)
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(50, 150)
        self.vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
        self.lifespan = random.uniform(0.5, 1.5)
        self.color = color
        self.radius = random.uniform(2, 5)

    def update(self, dt):
        self.pos += self.vel * dt
        self.lifespan -= dt
        self.vel *= 0.95 # damping

    def draw(self, surface):
        if self.lifespan > 0:
            alpha = int(255 * (self.lifespan / 1.5))
            pygame.gfxdraw.filled_circle(surface, int(self.pos.x), int(self.pos.y), int(self.radius), (self.color.r, self.color.g, self.color.b, alpha))


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    game_description = "Collapse quantum cards to a stable state. Entangle adjacent cards to create pairs and try to form the target binary sequence."
    user_guide = "Use arrow keys to move the cursor. Press space to select a card, move to an adjacent card, and release space to entangle and collapse them. Hold shift and use arrow keys to change gravity."
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.W, self.H = 640, 400
        self.FPS = 30

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.H, self.W, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.W, self.H))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_card = pygame.font.SysFont("monospace", 12)
        self.font_title = pygame.font.SysFont("monospace", 24, bold=True)

        # Colors
        self.COLOR_BG = pygame.Color("#0a0f2d")
        self.COLOR_GRID = pygame.Color("#1f295c")
        self.COLOR_CURSOR = pygame.Color("#ffffff")
        self.COLOR_TEXT = pygame.Color("#e0e0ff")
        self.COLOR_ZERO = pygame.Color("#4d94ff") # Blue
        self.COLOR_ONE = pygame.Color("#ff4d4d")  # Red
        self.COLOR_WIN = pygame.Color("#4dff4d") # Green
        self.COLOR_FAIL = pygame.Color("#ff4d4d") # Red
        self.COLOR_TARGET_BOX = pygame.Color("#2a346e")

        # Game state variables
        self.grid = []
        self.grid_size = (3, 3)
        self.target_state = []
        self.target_bit_length = 2
        self.cursor_pos = [0, 0]
        self.gravity_dir = 1 # 1:up, 2:down, 3:left, 4:right
        self.selection1 = None
        self.prev_space_held = False
        self.particles = []
        self.episodes_completed = 0
        self.max_steps = 1000

        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_over_message = ""
        self.game_over_timer = 0
        
        # Difficulty scaling
        if self.episodes_completed > 0 and self.episodes_completed % 5 == 0:
            rows, cols = self.grid_size
            if rows < 5: self.grid_size = (rows + 1, cols + 1)
        if self.episodes_completed > 0 and self.episodes_completed % 10 == 0:
            if self.target_bit_length < self.grid_size[0] * self.grid_size[1]:
                 self.target_bit_length += 1
        
        self.grid = [
            [QuantumCard(r, c, self.grid_size, (self.W, self.H), self.np_random) for c in range(self.grid_size[1])]
            for r in range(self.grid_size[0])
        ]
        
        self.target_state = [self.np_random.integers(0, 2) for _ in range(self.target_bit_length)]
        
        self.cursor_pos = [self.grid_size[0] // 2, self.grid_size[1] // 2]
        self.selection1 = None
        self.prev_space_held = False
        self.particles = []

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        terminated = False
        dt = 1.0 / self.FPS
        
        if self.game_over:
            self.game_over_timer -= dt
            if self.game_over_timer <= 0:
                terminated = True
            return self._get_observation(), 0, terminated, False, self._get_info()

        movement, space_held, shift_held = action
        
        action_taken = False

        # --- Handle Input ---
        if shift_held:
            if movement > 0:
                self.gravity_dir = movement
                action_taken = True
        else:
            moved = False
            if movement == 1 and self.cursor_pos[0] > 0: self.cursor_pos[0] -= 1; moved = True
            elif movement == 2 and self.cursor_pos[0] < self.grid_size[0] - 1: self.cursor_pos[0] += 1; moved = True
            elif movement == 3 and self.cursor_pos[1] > 0: self.cursor_pos[1] -= 1; moved = True
            elif movement == 4 and self.cursor_pos[1] < self.grid_size[1] - 1: self.cursor_pos[1] += 1; moved = True
            if moved: action_taken = True

        # Space press/release for matching
        space_pressed = space_held and not self.prev_space_held
        space_released = not space_held and self.prev_space_held

        if space_pressed:
            r, c = self.cursor_pos
            if self.grid[r][c].is_superposition:
                self.selection1 = (r, c)
                action_taken = True

        if space_released and self.selection1:
            r1, c1 = self.selection1
            r2, c2 = self.cursor_pos
            
            # Check for adjacency
            if (r1, c1) != (r2, c2) and abs(r1 - r2) + abs(c1 - c2) == 1:
                card1 = self.grid[r1][c1]
                card2 = self.grid[r2][c2]
                if card1.is_superposition and card2.is_superposition:
                    action_taken = True
                    # Perform collapse
                    p0 = (card1.alpha**2 + card2.alpha**2) / 2
                    outcome = 0 if self.np_random.random() < p0 else 1
                    
                    card1.collapse(outcome)
                    card2.collapse(outcome)
                    
                    reward += 1.0
                    self.score += 10
                    
                    # Create particles
                    pos1 = card1.render_pos + pygame.Vector2(card1.size/2, card1.size/2)
                    pos2 = card2.render_pos + pygame.Vector2(card2.size/2, card2.size/2)
                    mid_pos = (pos1 + pos2) / 2
                    for _ in range(30):
                        self.particles.append(Particle(mid_pos.x, mid_pos.y, self.COLOR_WIN if outcome == 0 else self.COLOR_ONE))

            self.selection1 = None

        if not action_taken and movement == 0:
            reward -= 0.1

        self.prev_space_held = space_held

        # --- Update Game Logic ---
        self.steps += 1
        for row in self.grid:
            for card in row:
                card.update(dt)
        
        for p in self.particles:
            p.update(dt)
        self.particles = [p for p in self.particles if p.lifespan > 0]

        # --- Check Termination ---
        all_collapsed = all(not card.is_superposition for row in self.grid for card in row)
        
        if all_collapsed:
            self.game_over = True
            current_state = []
            flat_grid = [card for row in self.grid for card in row]
            for card in flat_grid:
                current_state.append(card.collapsed_state)
            
            # Check if current state contains the target state
            is_win = False
            if len(current_state) >= len(self.target_state):
                for i in range(len(current_state) - len(self.target_state) + 1):
                    if current_state[i:i+len(self.target_state)] == self.target_state:
                        is_win = True
                        break
            
            if is_win:
                reward += 100
                self.score += 1000
                self.game_over_message = "TARGET STATE ACHIEVED"
                self.game_over_color = self.COLOR_WIN
                self.episodes_completed += 1
            else:
                reward -= 100
                self.score -= 100
                self.game_over_message = "STATE MISMATCH"
                self.game_over_color = self.COLOR_FAIL
            
            self.game_over_timer = 2.0 # Show message for 2 seconds

        if self.steps >= self.max_steps and not self.game_over:
            self.game_over = True
            self.game_over_message = "MAX STEPS REACHED"
            self.game_over_color = self.COLOR_FAIL
            self.game_over_timer = 2.0
            reward -= 100

        if self.game_over and self.game_over_timer <= 0:
             terminated = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _draw_glow_rect(self, surface, rect, color, glow_size=10):
        for i in range(glow_size, 0, -1):
            alpha = int(100 * (1 - i / glow_size))
            glow_rect = rect.inflate(i*2, i*2)
            s = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(s, (color.r, color.g, color.b, alpha), s.get_rect(), border_radius=8)
            surface.blit(s, glow_rect.topleft)

    def _render_game(self):
        # Draw grid lines
        for r in range(self.grid_size[0]):
            for c in range(self.grid_size[1]):
                card = self.grid[r][c]
                rect = pygame.Rect(card.target_pos.x, card.target_pos.y, card.size, card.size)
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1, border_radius=8)

        # Draw cards
        for r in range(self.grid_size[0]):
            for c in range(self.grid_size[1]):
                card = self.grid[r][c]
                rect = pygame.Rect(card.render_pos.x, card.render_pos.y, card.size, card.size)
                
                # Draw glow for selection
                if self.selection1 == (r, c):
                    self._draw_glow_rect(self.screen, rect, self.COLOR_CURSOR, 10)

                # Draw card body
                pygame.draw.rect(self.screen, card.color, rect, border_radius=8)
                pygame.draw.rect(self.screen, self.COLOR_TEXT, rect, 1, border_radius=8)
                
                # Draw card text
                center = (rect.centerx, rect.centery)
                if card.is_superposition:
                    text = f"{card.alpha:.2f}|0> + {card.beta:.2f}|1>"
                    draw_text(self.screen, text, (center[0], center[1]), self.font_card, self.COLOR_TEXT)
                else:
                    text = f"|{card.collapsed_state}>"
                    font_size = int(lerp(40, 30, card.anim_progress)) if card.anim_state == 'collapse' else 30
                    temp_font = pygame.font.SysFont("monospace", font_size, bold=True)
                    draw_text(self.screen, text, center, temp_font, self.COLOR_TEXT)
        
        # Draw cursor
        cursor_r, cursor_c = self.cursor_pos
        card = self.grid[cursor_r][cursor_c]
        cursor_rect = pygame.Rect(card.target_pos.x - 5, card.target_pos.y - 5, card.size + 10, card.size + 10)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 2, border_radius=10)

        # Draw particles
        for p in self.particles:
            p.draw(self.screen)
            
    def _render_ui(self):
        # Score and Steps
        draw_text(self.screen, f"SCORE: {self.score}", (10, 10), self.font_main, self.COLOR_TEXT, center=False)
        draw_text(self.screen, f"STEP: {self.steps}/{self.max_steps}", (self.W - 150, 10), self.font_main, self.COLOR_TEXT, center=False)
        
        # Target State
        target_box_w = len(self.target_state) * 25 + 10
        target_box_rect = pygame.Rect((self.W - target_box_w) / 2, 20, target_box_w, 40)
        pygame.draw.rect(self.screen, self.COLOR_TARGET_BOX, target_box_rect, border_radius=8)
        draw_text(self.screen, "TARGET", (self.W / 2, 15), self.font_card, self.COLOR_TEXT)
        for i, bit in enumerate(self.target_state):
            color = self.COLOR_ZERO if bit == 0 else self.COLOR_ONE
            pos_x = target_box_rect.left + 15 + i * 25
            pygame.draw.circle(self.screen, color, (pos_x, target_box_rect.centery), 8)

        # Gravity Indicator
        grav_pos = pygame.Vector2(self.W - 40, self.H - 40)
        pygame.draw.circle(self.screen, self.COLOR_GRID, grav_pos, 25, 2)
        draw_text(self.screen, "GRAVITY", (grav_pos.x, grav_pos.y - 35), self.font_card, self.COLOR_TEXT)
        
        angle = 0
        if self.gravity_dir == 1: angle = math.pi # Up
        if self.gravity_dir == 2: angle = 0 # Down
        if self.gravity_dir == 3: angle = math.pi / 2 # Left
        if self.gravity_dir == 4: angle = -math.pi / 2 # Right
        
        p1 = grav_pos + pygame.Vector2(15, 0).rotate_rad(angle)
        p2 = grav_pos + pygame.Vector2(-8, 8).rotate_rad(angle)
        p3 = grav_pos + pygame.Vector2(-8, -8).rotate_rad(angle)
        pygame.gfxdraw.aapolygon(self.screen, [p1, p2, p3], self.COLOR_CURSOR)
        pygame.gfxdraw.filled_polygon(self.screen, [p1, p2, p3], self.COLOR_CURSOR)
        
        # Game Over Message
        if self.game_over:
            s = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
            alpha = min(200, int(200 * (2.0 - self.game_over_timer)))
            s.fill((0, 0, 0, alpha))
            self.screen.blit(s, (0, 0))
            draw_text(self.screen, self.game_over_message, (self.W/2, self.H/2), self.font_title, self.game_over_color)

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
            "cursor_pos": self.cursor_pos,
            "all_collapsed": all(not card.is_superposition for row in self.grid for card in row),
        }

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.W, env.H))
    pygame.display.set_caption("Quantum Collapse")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0
        space_held = 0
        shift_held = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Draw the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished. Total reward: {total_reward:.2f}. Score: {info['score']}")
            obs, info = env.reset()
            total_reward = 0
            
        clock.tick(env.FPS)
        
    env.close()