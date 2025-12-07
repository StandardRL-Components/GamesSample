
# Generated: 2025-08-28T03:13:55.596123
# Source Brief: brief_04856.md
# Brief Index: 4856

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press space to swipe and slice fruit. Avoid bombs!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced arcade game where you slice falling fruit with a virtual blade. Score points for each fruit sliced, create combos for extra points, but be careful not to slice the bombs or miss too many fruits."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    BLADE_SPEED = 10
    SWIPE_LENGTH = 150
    SWIPE_DURATION = 4  # frames
    FRUIT_RADIUS = 15
    BOMB_RADIUS = 18
    MAX_LIVES = 5
    WIN_SCORE = 30
    MAX_STEPS = 1000
    
    # --- Colors ---
    COLOR_BG_1 = (40, 42, 54)
    COLOR_BG_2 = (60, 62, 74)
    COLOR_BLADE = (255, 255, 255)
    COLOR_TRAIL = (255, 255, 100)
    COLOR_TEXT = (248, 248, 242)
    COLOR_TEXT_SHADOW = (30, 30, 40)
    COLOR_BOMB = (80, 80, 80)
    COLOR_BOMB_FUSE = (255, 100, 100)
    
    FRUIT_COLORS = [
        {'main': (139, 233, 253), 'splash': (200, 255, 255)}, # Cyan
        {'main': (80, 250, 123), 'splash': (150, 255, 180)}, # Green
        {'main': (255, 184, 108), 'splash': (255, 220, 150)}, # Orange
        {'main': (255, 121, 198), 'splash': (255, 180, 220)}, # Pink
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
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
        self.font_large = pygame.font.SysFont("Consolas", 30, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_combo = pygame.font.SysFont("Verdana", 24, bold=True)

        # Etc...
        self.game_objects = None
        
        # Initialize state variables
        self.reset()
        
        # self.validate_implementation() # Optional: Uncomment to run validation on init

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.game_objects = {
            'blade_pos': np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=np.float32),
            'last_move_dir': np.array([1.0, 0.0], dtype=np.float32),
            'last_space_held': False,
            'swipes': [],
            'fruits': [],
            'bombs': [],
            'particles': [],
            'sliced_pieces': [],
            'combo_texts': [],
            'score': 0,
            'fruits_sliced_total': 0,
            'lives': self.MAX_LIVES,
            'steps': 0,
            'game_over': False,
            'win': False,
            'fruit_fall_speed': 2.0,
            'spawn_timer': 0,
            'spawn_interval': 40,
        }
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def step(self, action):
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        # shift_held is unused per brief
        
        reward = 0.0
        
        # --- Handle Input and Blade Movement ---
        move_vector = np.array([0.0, 0.0])
        if movement == 1: move_vector[1] = -1
        elif movement == 2: move_vector[1] = 1
        elif movement == 3: move_vector[0] = -1
        elif movement == 4: move_vector[0] = 1

        if np.any(move_vector):
            self.game_objects['last_move_dir'] = move_vector
        
        self.game_objects['blade_pos'] += move_vector * self.BLADE_SPEED
        self.game_objects['blade_pos'][0] = np.clip(self.game_objects['blade_pos'][0], 0, self.WIDTH)
        self.game_objects['blade_pos'][1] = np.clip(self.game_objects['blade_pos'][1], 0, self.HEIGHT)

        # --- Handle Swipe Action ---
        swipe_triggered = space_held and not self.game_objects['last_space_held']
        if swipe_triggered:
            # sfx: blade_swipe.wav
            start_pos = self.game_objects['blade_pos'].copy()
            end_pos = start_pos + self.game_objects['last_move_dir'] * self.SWIPE_LENGTH
            self.game_objects['swipes'].append({
                'start': start_pos, 'end': end_pos, 'life': self.SWIPE_DURATION
            })
            
            sliced_in_swipe = self._process_swipe(start_pos, end_pos)
            
            # Grant rewards for slices
            reward += sliced_in_swipe['fruits']
            if sliced_in_swipe['fruits'] > 1:
                # sfx: combo_bonus.wav
                reward += 5.0 # Combo bonus
                self._create_combo_text(sliced_in_swipe['fruits'], start_pos)

            if sliced_in_swipe['bombs'] > 0:
                # sfx: explosion.wav
                reward -= 5.0
                self.game_objects['lives'] = 0 # Slicing a bomb is instant game over
                self.game_objects['game_over'] = True

        self.game_objects['last_space_held'] = space_held

        # --- Update Game State (if not over) ---
        if not self.game_objects['game_over']:
            self._update_objects()
            self._spawn_objects()
        
        # --- Update Timers and Counters ---
        self.steps = self.game_objects['steps']
        self.steps += 1
        self.game_objects['steps'] = self.steps
        if self.steps % 50 == 0:
            self.game_objects['fruit_fall_speed'] = min(8.0, self.game_objects['fruit_fall_speed'] + 0.05)
            self.game_objects['spawn_interval'] = max(15, self.game_objects['spawn_interval'] - 1)

        # --- Check for Termination and Calculate Reward ---
        terminated, term_reward = self._check_termination()
        reward += term_reward
        self.score = self.game_objects['score']
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _process_swipe(self, p1, p2):
        sliced_count = {'fruits': 0, 'bombs': 0}
        
        for fruit in self.game_objects['fruits'][:]:
            if self._line_circle_intersect(p1, p2, fruit['pos'], self.FRUIT_RADIUS):
                # sfx: fruit_slice.wav
                self.game_objects['fruits'].remove(fruit)
                self._create_slice_effect(fruit['pos'], self.game_objects['last_move_dir'], fruit['color'])
                self.game_objects['score'] += 1
                self.game_objects['fruits_sliced_total'] += 1
                sliced_count['fruits'] += 1

        for bomb in self.game_objects['bombs'][:]:
            if self._line_circle_intersect(p1, p2, bomb['pos'], self.BOMB_RADIUS):
                self.game_objects['bombs'].remove(bomb)
                self._create_explosion_effect(bomb['pos'])
                sliced_count['bombs'] += 1

        return sliced_count
    
    def _update_objects(self):
        # Update fruits
        for fruit in self.game_objects['fruits'][:]:
            fruit['pos'] += fruit['vel']
            if fruit['pos'][1] > self.HEIGHT + self.FRUIT_RADIUS:
                # sfx: life_lost.wav
                self.game_objects['fruits'].remove(fruit)
                self.game_objects['lives'] -= 1
        
        # Update bombs
        for bomb in self.game_objects['bombs'][:]:
            bomb['pos'] += bomb['vel']
            if bomb['pos'][1] > self.HEIGHT + self.BOMB_RADIUS:
                self.game_objects['bombs'].remove(bomb)

        # Update swipes
        for swipe in self.game_objects['swipes'][:]:
            swipe['life'] -= 1
            if swipe['life'] <= 0:
                self.game_objects['swipes'].remove(swipe)

        # Update particles
        for p in self.game_objects['particles'][:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.game_objects['particles'].remove(p)
        
        # Update sliced pieces
        for piece in self.game_objects['sliced_pieces'][:]:
            piece['pos'] += piece['vel']
            piece['vel'][1] += 0.2 # Gravity
            piece['angle'] += piece['rot']
            piece['life'] -= 1
            if piece['life'] <= 0:
                self.game_objects['sliced_pieces'].remove(piece)

        # Update combo texts
        for text in self.game_objects['combo_texts'][:]:
            text['pos'][1] -= 0.5
            text['life'] -= 1
            if text['life'] <= 0:
                self.game_objects['combo_texts'].remove(text)

    def _spawn_objects(self):
        self.game_objects['spawn_timer'] += 1
        if self.game_objects['spawn_timer'] >= self.game_objects['spawn_interval']:
            self.game_objects['spawn_timer'] = 0
            
            x_pos = self.np_random.uniform(self.FRUIT_RADIUS * 2, self.WIDTH - self.FRUIT_RADIUS * 2)
            y_pos = -self.FRUIT_RADIUS
            vx = self.np_random.uniform(-1.5, 1.5)
            vy = self.game_objects['fruit_fall_speed']

            if self.np_random.random() < 0.15: # 15% chance for a bomb
                self.game_objects['bombs'].append({
                    'pos': np.array([x_pos, y_pos], dtype=np.float32),
                    'vel': np.array([vx, vy], dtype=np.float32),
                })
            else:
                color_choice_index = self.np_random.integers(0, len(self.FRUIT_COLORS))
                self.game_objects['fruits'].append({
                    'pos': np.array([x_pos, y_pos], dtype=np.float32),
                    'vel': np.array([vx, vy], dtype=np.float32),
                    'color': self.FRUIT_COLORS[color_choice_index],
                })
    
    def _check_termination(self):
        if self.game_objects['game_over']:
            return True, -50.0

        if self.game_objects['lives'] <= 0:
            self.game_objects['game_over'] = True
            return True, -50.0
        
        if self.game_objects['fruits_sliced_total'] >= self.WIN_SCORE:
            self.game_objects['game_over'] = True
            self.game_objects['win'] = True
            return True, 50.0

        if self.game_objects['steps'] >= self.MAX_STEPS:
            self.game_objects['game_over'] = True
            return True, 0.0

        return False, 0.0

    def _get_observation(self):
        # Clear screen with background gradient
        for y in range(self.HEIGHT):
            interp = y / self.HEIGHT
            color = [int(c1 * (1 - interp) + c2 * interp) for c1, c2 in zip(self.COLOR_BG_1, self.COLOR_BG_2)]
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw fruits
        for fruit in self.game_objects['fruits']:
            pos = fruit['pos'].astype(int)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.FRUIT_RADIUS, fruit['color']['main'])
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.FRUIT_RADIUS, fruit['color']['main'])

        # Draw bombs
        for bomb in self.game_objects['bombs']:
            pos = bomb['pos'].astype(int)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.BOMB_RADIUS, self.COLOR_BOMB)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.BOMB_RADIUS, self.COLOR_BOMB)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.BOMB_RADIUS-2, (120,120,120))
            pygame.draw.circle(self.screen, self.COLOR_BOMB_FUSE, (pos[0] + 8, pos[1] - 8), 3)

        # Draw sliced pieces
        for piece in self.game_objects['sliced_pieces']:
            alpha = int(255 * (piece['life'] / piece['max_life']))
            if alpha > 0:
                piece['surf'].set_alpha(alpha)
                rotated_surf = pygame.transform.rotate(piece['surf'], piece['angle'])
                rect = rotated_surf.get_rect(center=piece['pos'].astype(int))
                self.screen.blit(rotated_surf, rect)

        # Draw particles
        for p in self.game_objects['particles']:
            alpha = max(0, 255 * (p['life'] / p['max_life']))
            color = (*p['color'], alpha)
            temp_surf = pygame.Surface((p['radius']*2, p['radius']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p['radius'], p['radius']), p['radius'])
            self.screen.blit(temp_surf, p['pos'].astype(int) - p['radius'])

        # Draw swipe trails
        for swipe in self.game_objects['swipes']:
            alpha = max(0, int(255 * (swipe['life'] / self.SWIPE_DURATION)))
            pygame.draw.line(self.screen, (*self.COLOR_TRAIL, alpha), swipe['start'], swipe['end'], width=10)

        # Draw blade cursor
        blade_pos = self.game_objects['blade_pos'].astype(int)
        pygame.gfxdraw.aacircle(self.screen, blade_pos[0], blade_pos[1], 10, self.COLOR_BLADE)
        pygame.gfxdraw.filled_circle(self.screen, blade_pos[0], blade_pos[1], 10, self.COLOR_BLADE)
        pygame.gfxdraw.aacircle(self.screen, blade_pos[0], blade_pos[1], 12, (255, 255, 255, 100))
        
    def _render_ui(self):
        # Draw Score
        score_text = f"SCORE: {self.game_objects['score']}"
        self._draw_text(score_text, self.font_large, (10, 5))
        
        # Draw Lives
        for i in range(self.game_objects['lives']):
            pos_x = self.WIDTH - 30 - (i * (self.FRUIT_RADIUS + 5))
            pygame.gfxdraw.aacircle(self.screen, pos_x, 20, self.FRUIT_RADIUS // 2, self.FRUIT_COLORS[1]['main'])
            pygame.gfxdraw.filled_circle(self.screen, pos_x, 20, self.FRUIT_RADIUS // 2, self.FRUIT_COLORS[1]['main'])

        # Draw combo texts
        for text_info in self.game_objects['combo_texts']:
            alpha = int(255 * (text_info['life'] / text_info['max_life']))
            if alpha > 0:
                self._draw_text(text_info['text'], self.font_combo, text_info['pos'], alpha=alpha)

        # Draw Game Over / Win message
        if self.game_objects['game_over']:
            message = "YOU WIN!" if self.game_objects['win'] else "GAME OVER"
            self._draw_text(message, self.font_large, 
                            (self.WIDTH // 2, self.HEIGHT // 2 - 20), center=True)

    def _draw_text(self, text, font, pos, color=COLOR_TEXT, shadow_color=COLOR_TEXT_SHADOW, center=False, alpha=255):
        text_surf = font.render(text, True, color)
        shadow_surf = font.render(text, True, shadow_color)
        text_surf.set_alpha(alpha)
        shadow_surf.set_alpha(alpha)
        
        text_rect = text_surf.get_rect()
        if center:
            text_rect.center = pos
        else:
            text_rect.topleft = pos
        
        shadow_rect = text_rect.copy()
        shadow_rect.x += 2
        shadow_rect.y += 2
        
        self.screen.blit(shadow_surf, shadow_rect)
        self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.game_objects['score'],
            "steps": self.game_objects['steps'],
        }

    # --- Effect Creation ---
    def _create_slice_effect(self, pos, direction, color_info):
        # Create two half-fruit pieces
        perp_dir = np.array([-direction[1], direction[0]]) # Perpendicular
        for i in [-1, 1]:
            piece_surf = pygame.Surface((self.FRUIT_RADIUS * 2, self.FRUIT_RADIUS * 2), pygame.SRCALPHA)
            pygame.draw.circle(piece_surf, color_info['main'], (self.FRUIT_RADIUS, self.FRUIT_RADIUS), self.FRUIT_RADIUS)
            clip_rect = pygame.Rect(0, 0, self.FRUIT_RADIUS, self.FRUIT_RADIUS * 2) if i == -1 else pygame.Rect(self.FRUIT_RADIUS, 0, self.FRUIT_RADIUS, self.FRUIT_RADIUS * 2)
            piece_surf.fill((0,0,0,0), clip_rect)

            self.game_objects['sliced_pieces'].append({
                'pos': pos.copy(),
                'vel': (perp_dir * i * self.np_random.uniform(2, 4)) + np.array([0, -3]),
                'surf': piece_surf,
                'angle': 0,
                'rot': self.np_random.uniform(-15, 15),
                'life': 40,
                'max_life': 40
            })
        
        # Create splash particles
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            life = self.np_random.integers(15, 30)
            color_index = self.np_random.integers(0, len(self.FRUIT_COLORS))
            self.game_objects['particles'].append({
                'pos': pos.copy(), 'vel': vel, 'radius': self.np_random.integers(2, 5),
                'color': color_info['splash'], 'life': life, 'max_life': life
            })

    def _create_explosion_effect(self, pos):
        # Big flash
        self.game_objects['particles'].append({
            'pos': pos - 50, 'vel': np.zeros(2), 'radius': 50,
            'color': (255, 255, 200), 'life': 3, 'max_life': 3
        })
        # Fiery particles
        for _ in range(50):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(2, 8)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            life = self.np_random.integers(20, 40)
            color_options = [(255, 100, 0), (255, 200, 0), (200, 200, 200)]
            color = color_options[self.np_random.integers(0, len(color_options))]
            self.game_objects['particles'].append({
                'pos': pos.copy(), 'vel': vel, 'radius': self.np_random.integers(2, 6),
                'color': color,
                'life': life, 'max_life': life
            })

    def _create_combo_text(self, count, pos):
        self.game_objects['combo_texts'].append({
            'text': f"x{count} COMBO!",
            'pos': pos.copy() - np.array([40, 50]),
            'life': 45,
            'max_life': 45,
        })

    # --- Utility ---
    def _line_circle_intersect(self, p1, p2, circle_center, r):
        # Using vector math to find the closest point on the line segment to the circle center
        d = p2 - p1
        if np.all(d == 0): # The "line" is a point
            return np.linalg.norm(p1 - circle_center) <= r

        t = np.dot(circle_center - p1, d) / np.dot(d, d)
        t = np.clip(t, 0, 1)
        closest_point = p1 + t * d
        dist = np.linalg.norm(circle_center - closest_point)
        return dist <= r

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        print("✓ Running implementation validation...")
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

# Example usage for testing and playing the game
if __name__ == '__main__':
    # Set this to run in a headless environment if no display is available
    # import os
    # os.environ["SDL_VIDEODRIVER"] = "dummy"

    env = GameEnv()
    env.validate_implementation()
    
    obs, info = env.reset()
    
    # To run the game live, a display is needed
    try:
        screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
        pygame.display.set_caption("Fruit Slicer")
        clock = pygame.time.Clock()
        
        running = True
        total_reward = 0.0
        
        while running:
            movement = 0 # none
            space_held = 0
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False

            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            if keys[pygame.K_SPACE]: space_held = 1
            
            action = [movement, space_held, 0] # shift is unused
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            # Render the observation to the display
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()

            if terminated:
                print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
                total_reward = 0.0
                # Wait a bit before resetting
                pygame.time.wait(2000)
                obs, info = env.reset()

            clock.tick(30) # Run at 30 FPS

    except pygame.error as e:
        print(f"Pygame display error: {e}")
        print("Cannot run live game without a display. The environment is still valid for headless training.")

    finally:
        env.close()