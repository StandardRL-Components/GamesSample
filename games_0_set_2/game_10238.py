import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:09:24.939412
# Source Brief: brief_00238.md
# Brief Index: 238
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "A platforming puzzle game. Jump around, collect numbers and operators, "
        "and place them to solve mathematical equations."
    )
    user_guide = (
        "Controls: ←→ to move, ↑ to jump. Press space to pick up or place a symbol. "
        "Press shift to clear the current equation."
    )
    auto_advance = True

    # --- CONSTANTS ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    PLAYER_SIZE = 20
    PLAYER_ACCEL = 1.2
    PLAYER_FRICTION = 0.85
    PLAYER_MAX_SPEED = 6
    GRAVITY = 0.8
    JUMP_STRENGTH = -14

    # --- COLORS ---
    COLOR_BG = (15, 20, 30)
    COLOR_PLATFORM = (60, 80, 110)
    COLOR_PLATFORM_OUTLINE = (90, 120, 160)
    COLOR_PLAYER = (255, 215, 0)
    COLOR_PLAYER_GLOW = (255, 215, 0, 50)
    COLOR_SYMBOL_BG = (255, 255, 255, 20)
    COLOR_SYMBOL_TEXT = (240, 240, 240)
    COLOR_UI_TEXT = (200, 200, 220)
    COLOR_SUCCESS = (0, 255, 127)
    COLOR_FAIL = (255, 69, 0)
    COLOR_PLACEHOLDER = (100, 120, 150)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont('Consolas', 16, bold=True)
        self.font_medium = pygame.font.SysFont('Consolas', 24, bold=True)
        self.font_large = pygame.font.SysFont('Consolas', 32, bold=True)

        self.player_pos = None
        self.player_vel = None
        self.player_rect = None
        self.on_ground = False
        self.held_symbol = None
        self.prev_space_held = False
        self.prev_shift_held = False
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.platforms = []
        self.symbol_sources = []
        self.equations = []
        self.current_equation_idx = 0
        self.placed_symbols = []
        self.unlocked_operators = []
        self.particles = []

        self.reset()
        # self.validate_implementation(self) # Commented out for submission

    def _generate_level(self):
        self.platforms = [
            pygame.Rect(0, self.HEIGHT - 20, self.WIDTH, 20),
            pygame.Rect(100, 280, 120, 15),
            pygame.Rect(420, 280, 120, 15),
            pygame.Rect(250, 200, 140, 15),
            pygame.Rect(0, 150, 80, 15),
            pygame.Rect(self.WIDTH - 80, 150, 80, 15),
        ]

        self.symbol_sources = []
        symbols_to_place = list(set(
            [str(i) for i in range(1, 10)] + ['+', '-', '*', '/']
        ))
        random.shuffle(symbols_to_place)

        positions = [
            (150, 250), (470, 250), (310, 170),
            (30, 120), (self.WIDTH - 50, 120),
            (50, self.HEIGHT - 50), (150, self.HEIGHT - 50), (250, self.HEIGHT - 50),
            (350, self.HEIGHT - 50), (450, self.HEIGHT - 50), (550, self.HEIGHT - 50)
        ]
        for i, symbol in enumerate(symbols_to_place):
            if i < len(positions):
                pos = pygame.Vector2(positions[i])
                self.symbol_sources.append({
                    'symbol': symbol,
                    'pos': pos,
                    'rect': pygame.Rect(pos.x - 15, pos.y - 15, 30, 30)
                })

        self.equations = [
            {'slots': 3, 'target': 8, 'display': '_ + _ = 8'},
            {'slots': 3, 'target': 5, 'display': '_ - _ = 5'},
            {'slots': 3, 'target': 12, 'display': '_ * _ = 12'},
            {'slots': 5, 'target': 14, 'display': '_ * _ + _ = 14'},
            {'slots': 3, 'target': 3, 'display': '_ / _ = 3'},
            {'slots': 5, 'target': 1, 'display': '_ - _ * _ = 1'},
        ]
        self.unlocked_operators = ['+', '-']

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.score = 0
        self.game_over = False

        self.player_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT - 50)
        self.player_vel = pygame.Vector2(0, 0)
        self.player_rect = pygame.Rect(self.player_pos.x, self.player_pos.y, self.PLAYER_SIZE, self.PLAYER_SIZE)
        self.on_ground = False
        self.held_symbol = None
        self.prev_space_held = False
        self.prev_shift_held = False
        self.particles = []

        self._generate_level()
        self.current_equation_idx = 0
        self.placed_symbols = [None] * self.equations[0]['slots']

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = -0.01  # Small time penalty

        # --- Handle Input ---
        space_press = space_held and not self.prev_space_held
        shift_press = shift_held and not self.prev_shift_held

        # Movement
        if movement == 3:  # Left
            self.player_vel.x -= self.PLAYER_ACCEL
        elif movement == 4:  # Right
            self.player_vel.x += self.PLAYER_ACCEL
        if movement == 1 and self.on_ground:  # Jump
            self.player_vel.y = self.JUMP_STRENGTH
            self.on_ground = False
            # sfx: jump

        # Action: Reset Equation
        if shift_press:
            self.placed_symbols = [None] * len(self.placed_symbols)
            reward -= 0.5
            self._spawn_particles(self.WIDTH / 2, 60, 30, self.COLOR_FAIL, count=30)
            # sfx: reset_equation

        # Action: Clone/Place Symbol
        if space_press:
            if self.held_symbol:
                reward += self._action_place_symbol()
            else:
                reward += self._action_clone_symbol()

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        # --- Physics & Update ---
        self._update_player_physics()
        self._update_particles()
        
        # Check if equation is complete and evaluate
        if self.held_symbol is None and all(s is not None for s in self.placed_symbols):
            reward += self._evaluate_and_progress()

        # --- Termination ---
        self.steps += 1
        terminated = False
        if self.player_pos.y > self.HEIGHT + 50:
            terminated = True
            reward = -100
        elif self.current_equation_idx >= len(self.equations):
            terminated = True
            reward = 100
            self.score = len(self.equations)
        elif self.steps >= 2000:
            terminated = True
        
        self.game_over = terminated

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _action_place_symbol(self):
        try:
            first_empty_slot = self.placed_symbols.index(None)
            self.placed_symbols[first_empty_slot] = self.held_symbol
            self.held_symbol = None
            # sfx: place_symbol
            self._spawn_particles(self.player_pos.x, self.player_pos.y, 15, self.COLOR_UI_TEXT, count=15)
            return 1.0
        except ValueError: # No empty slots
            return -0.1

    def _action_clone_symbol(self):
        for source in self.symbol_sources:
            if self.player_rect.colliderect(source['rect']):
                if source['symbol'] in self.unlocked_operators or source['symbol'].isdigit():
                    self.held_symbol = source['symbol']
                    # sfx: clone_symbol
                    self._spawn_particles(source['pos'].x, source['pos'].y, 20, self.COLOR_PLAYER, count=20)
                    return 0.1
        return 0.0

    def _evaluate_and_progress(self):
        current_eq = self.equations[self.current_equation_idx]
        result = self._safe_eval(self.placed_symbols)
        
        if result is not None and math.isclose(result, current_eq['target']):
            # --- SUCCESS ---
            self.score += 1
            self.current_equation_idx += 1
            # sfx: success
            self._spawn_particles(self.WIDTH / 2, 60, 50, self.COLOR_SUCCESS, count=50, gravity=0.1)
            
            if self.current_equation_idx < len(self.equations):
                next_eq_slots = self.equations[self.current_equation_idx]['slots']
                self.placed_symbols = [None] * next_eq_slots
                # Unlock new operators
                if self.score == 2 and '*' not in self.unlocked_operators: self.unlocked_operators.append('*')
                if self.score == 4 and '/' not in self.unlocked_operators: self.unlocked_operators.append('/')
            return 5.0
        else:
            # --- FAIL ---
            # sfx: fail
            self._spawn_particles(self.WIDTH / 2, 60, 30, self.COLOR_FAIL, count=30)
            self.placed_symbols = [None] * len(self.placed_symbols)
            return -2.0
            
    def _safe_eval(self, tokens):
        if len(tokens) == 0: return None
        # Must be of form: num op num op num ...
        if len(tokens) % 2 == 0: return None

        allowed_tokens = self.unlocked_operators + [str(i) for i in range(10)]
        for token in tokens:
            if token not in allowed_tokens: return None

        try:
            # Simple left-to-right evaluation
            val = float(tokens[0])
            for i in range(1, len(tokens), 2):
                op, next_val_str = tokens[i], tokens[i+1]
                next_val = float(next_val_str)
                if op == '+': val += next_val
                elif op == '-': val -= next_val
                elif op == '*': val *= next_val
                elif op == '/':
                    if next_val == 0: return None # Division by zero
                    val /= next_val
            return val
        except (ValueError, IndexError):
            return None

    def _update_player_physics(self):
        # Apply friction
        self.player_vel.x *= self.PLAYER_FRICTION
        if abs(self.player_vel.x) < 0.1: self.player_vel.x = 0
        self.player_vel.x = max(-self.PLAYER_MAX_SPEED, min(self.PLAYER_MAX_SPEED, self.player_vel.x))

        # Apply gravity
        if not self.on_ground:
            self.player_vel.y += self.GRAVITY
        
        # Update position and check collisions
        self.player_pos.x += self.player_vel.x
        self.player_rect.x = int(self.player_pos.x)
        self._handle_collisions('horizontal')

        self.player_pos.y += self.player_vel.y
        self.player_rect.y = int(self.player_pos.y)
        self.on_ground = False
        self._handle_collisions('vertical')
        
        # Boundary checks
        if self.player_pos.x < 0:
            self.player_pos.x = 0
            self.player_vel.x = 0
        if self.player_pos.x > self.WIDTH - self.PLAYER_SIZE:
            self.player_pos.x = self.WIDTH - self.PLAYER_SIZE
            self.player_vel.x = 0

    def _handle_collisions(self, axis):
        for plat in self.platforms:
            if self.player_rect.colliderect(plat):
                if axis == 'horizontal':
                    if self.player_vel.x > 0: # Moving right
                        self.player_rect.right = plat.left
                    elif self.player_vel.x < 0: # Moving left
                        self.player_rect.left = plat.right
                    self.player_pos.x = self.player_rect.x
                    self.player_vel.x = 0
                elif axis == 'vertical':
                    if self.player_vel.y > 0: # Moving down
                        self.player_rect.bottom = plat.top
                        self.on_ground = True
                        self.player_vel.y = 0
                    elif self.player_vel.y < 0: # Moving up
                        self.player_rect.top = plat.bottom
                        self.player_vel.y = 0
                    self.player_pos.y = self.player_rect.y

    def _spawn_particles(self, x, y, radius, color, count=20, gravity=0.2):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 5)
            vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            self.particles.append({
                'pos': pygame.Vector2(x, y),
                'vel': vel,
                'lifespan': random.randint(15, 30),
                'color': color,
                'gravity': gravity
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'].y += p['gravity']
            p['lifespan'] -= 1
        self.particles = [p for p in self.particles if p['lifespan'] > 0]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Particles
        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / 30))
            color = p['color']
            if len(color) == 4:
                pygame.draw.circle(self.screen, (*color[:3], alpha), p['pos'], int(p['lifespan'] / 5))
            else:
                 pygame.draw.circle(self.screen, color, p['pos'], int(p['lifespan'] / 5))

        # Platforms
        for plat in self.platforms:
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, plat, border_radius=3)
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM_OUTLINE, plat, 1, border_radius=3)

        # Symbol Sources
        for source in self.symbol_sources:
            is_locked = source['symbol'] in ['*', '/'] and source['symbol'] not in self.unlocked_operators
            color = self.COLOR_PLACEHOLDER if is_locked else self.COLOR_SYMBOL_TEXT
            
            pygame.draw.rect(self.screen, self.COLOR_SYMBOL_BG, source['rect'], border_radius=5)
            text_surf = self.font_medium.render(source['symbol'], True, color)
            text_rect = text_surf.get_rect(center=source['rect'].center)
            self.screen.blit(text_surf, text_rect)

        # Player Glow
        glow_size = self.PLAYER_SIZE * 2.5
        glow_rect = pygame.Rect(0, 0, glow_size, glow_size)
        glow_rect.center = self.player_rect.center
        glow_surf = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, self.COLOR_PLAYER_GLOW, (glow_size/2, glow_size/2), glow_size/2)
        self.screen.blit(glow_surf, glow_rect)

        # Player
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, self.player_rect, border_radius=3)

        # Held Symbol
        if self.held_symbol:
            text_surf = self.font_medium.render(self.held_symbol, True, self.COLOR_PLAYER)
            text_rect = text_surf.get_rect(centerx=self.player_rect.centerx, bottom=self.player_rect.top - 5)
            self.screen.blit(text_surf, text_rect)

    def _render_ui(self):
        # --- Top UI: Equation ---
        if self.current_equation_idx < len(self.equations):
            eq = self.equations[self.current_equation_idx]
            
            # Display target equation
            eq_text = self.font_medium.render(eq['display'], True, self.COLOR_UI_TEXT)
            self.screen.blit(eq_text, eq_text.get_rect(centerx=self.WIDTH/2, top=15))

            # Display placed symbols
            total_width = eq['slots'] * 50
            start_x = self.WIDTH/2 - total_width/2
            for i, symbol in enumerate(self.placed_symbols):
                slot_rect = pygame.Rect(start_x + i * 50, 50, 40, 40)
                if symbol:
                    pygame.draw.rect(self.screen, self.COLOR_SYMBOL_BG, slot_rect, border_radius=5)
                    text_surf = self.font_large.render(symbol, True, self.COLOR_SYMBOL_TEXT)
                    self.screen.blit(text_surf, text_surf.get_rect(center=slot_rect.center))
                else:
                    pygame.draw.rect(self.screen, self.COLOR_PLACEHOLDER, slot_rect, 2, border_radius=5)

        # --- Bottom UI: Info ---
        score_text = f"Solved: {self.score} / {len(self.equations)}"
        score_surf = self.font_small.render(score_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(score_surf, (10, self.HEIGHT - 20))

        ops_text = f"Operators: {' '.join(self.unlocked_operators)}"
        ops_surf = self.font_small.render(ops_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(ops_surf, ops_surf.get_rect(right=self.WIDTH - 10, top=self.HEIGHT - 20))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "unlocked_operators": len(self.unlocked_operators),
            "held_symbol": self.held_symbol if self.held_symbol else "none"
        }
        
    def close(self):
        pygame.quit()

    @staticmethod
    def validate_implementation(self):
        print("Running implementation validation...")
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # --- Manual Play ---
    # Un-set the dummy driver to allow for display
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Symbolic Ascent")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # None
        space = 0 # Released
        shift = 0 # Released
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        if keys[pygame.K_UP]:
            movement = 1
        
        if keys[pygame.K_SPACE]:
            space = 1
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift = 1
            
        action = [movement, space, shift]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Rendering for manual play ---
        # The observation is already the rendered screen as a numpy array
        # We need to convert it back to a surface to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Episode finished! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            total_reward = 0
            obs, info = env.reset()
            # Add a small delay to show the final state
            pygame.time.wait(2000)

        clock.tick(GameEnv.FPS)
        
    env.close()