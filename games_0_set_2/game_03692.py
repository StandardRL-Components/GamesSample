
# Generated: 2025-08-28T00:07:50.327266
# Source Brief: brief_03692.md
# Brief Index: 3692

        
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
        "Controls: Use arrow keys to select an asteroid. Press space to confirm and move your ship to capture it."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "An isometric strategy game. You have 3 moves to capture 7 asteroids. More distant asteroids grant a higher reward."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 500
    
    NUM_ASTEROIDS = 12
    WIN_CONDITION_CAPTURES = 7
    MAX_MOVES = 3
    
    WORLD_SIZE = 120 # Logical grid size

    # Colors
    COLOR_BG = (10, 15, 30)
    COLOR_STAR = (200, 200, 220)
    COLOR_PLAYER = (0, 200, 255)
    COLOR_PLAYER_TRAIL = (0, 200, 255, 100)
    COLOR_ASTEROID_NEUTRAL = (120, 120, 140)
    COLOR_ASTEROID_CAPTURED = (0, 100, 150)
    COLOR_ASTEROID_SELECTED = (255, 255, 0)
    COLOR_TEXT = (255, 255, 255)
    COLOR_UI_BG = (20, 30, 50, 180)
    COLOR_GAMEOVER_WIN = (0, 255, 128)
    COLOR_GAMEOVER_LOSE = (255, 80, 80)

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
        self.font_small = pygame.font.SysFont('monospace', 16, bold=True)
        self.font_large = pygame.font.SysFont('monospace', 48, bold=True)
        
        self.stars = []
        self.asteroids = []
        self.player_ship = {}
        self.particles = []
        
        self.reset()
        
        # Run validation
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.moves_left = self.MAX_MOVES
        self.captured_count = 0
        self.game_over = False
        self.win = False
        
        self._generate_stars()
        self._generate_asteroids()
        
        self.player_ship = {
            'pos': np.array([self.WORLD_SIZE / 2, self.WORLD_SIZE / 2]),
            'prev_pos': np.array([self.WORLD_SIZE / 2, self.WORLD_SIZE / 2]),
            'screen_pos': self._project(self.WORLD_SIZE / 2, self.WORLD_SIZE / 2)
        }
        
        self.selected_asteroid_idx = None
        self.last_move_path = None
        self.particles = []
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = -0.01 # Small cost for existing
        
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        # 1. Handle Selection
        if movement != 0:
            self._handle_selection(movement)
            if self.selected_asteroid_idx is not None:
                dist_rank = self.asteroids[self.selected_asteroid_idx]['dist_rank']
                if dist_rank > 0.75: reward += 1.0
                else: reward -= 0.2

        # 2. Handle Confirmation
        elif space_held and self.selected_asteroid_idx is not None:
            target_idx = self.selected_asteroid_idx
            target_asteroid = self.asteroids[target_idx]

            if target_asteroid['status'] == 'neutral':
                # Store previous state for rendering trail
                self.player_ship['prev_pos'] = self.player_ship['pos'].copy()
                
                # Execute the move
                self.moves_left -= 1
                self.player_ship['pos'] = target_asteroid['pos'].copy()
                target_asteroid['status'] = 'captured'
                
                # Visuals
                self._create_capture_particles(target_asteroid['screen_pos'])
                # SFX: player.play_sound('capture')
                self.last_move_path = (self._project(*self.player_ship['prev_pos']), self._project(*self.player_ship['pos']))
                
                # Calculate rewards
                capture_reward = 5.0
                dist_rank = target_asteroid['dist_rank']
                if dist_rank > 0.75: capture_reward += 2.0
                elif dist_rank < 0.25: capture_reward -= 1.0
                reward += capture_reward
                
                self.captured_count += 1
                self.selected_asteroid_idx = None
            else:
                # SFX: player.play_sound('error')
                reward -= 0.5 # Penalty for trying to capture an invalid target
        
        # 3. Update game state
        self._update_particles()
        self.score += reward
        
        # 4. Check for Termination
        terminated = False
        if self.captured_count >= self.WIN_CONDITION_CAPTURES:
            terminated = True
            self.game_over = True
            self.win = True
            self.score += 50.0
            # SFX: player.play_sound('win')
        elif self.moves_left <= 0:
            terminated = True
            self.game_over = True
            self.win = False
            self.score += -50.0
            # SFX: player.play_sound('lose')
        
        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _project(self, x, y):
        iso_x = (x - y) * 1.5 + self.SCREEN_WIDTH / 2
        iso_y = (x + y) * 0.75 + self.SCREEN_HEIGHT / 2 - self.WORLD_SIZE
        return int(iso_x), int(iso_y)

    def _generate_stars(self):
        self.stars = []
        for _ in range(100):
            self.stars.append(
                (
                    self.np_random.integers(0, self.SCREEN_WIDTH),
                    self.np_random.integers(0, self.SCREEN_HEIGHT),
                    self.np_random.integers(1, 3)
                )
            )

    def _generate_asteroids(self):
        self.asteroids = []
        start_pos = np.array([self.WORLD_SIZE / 2, self.WORLD_SIZE / 2])
        
        for i in range(self.NUM_ASTEROIDS):
            while True:
                pos = self.np_random.uniform(low=10, high=self.WORLD_SIZE-10, size=2)
                if all(np.linalg.norm(pos - a['pos']) > 15 for a in self.asteroids):
                    break
            
            dist = np.linalg.norm(pos - start_pos)
            screen_pos = self._project(*pos)
            
            self.asteroids.append({
                'pos': pos,
                'screen_pos': screen_pos,
                'status': 'neutral',
                'distance': dist,
                'size': self.np_random.integers(5, 9)
            })
        
        # Calculate distance ranks for rewards
        distances = [a['distance'] for a in self.asteroids]
        sorted_distances = sorted(distances)
        for asteroid in self.asteroids:
            rank = sorted_distances.index(asteroid['distance']) / (len(sorted_distances) - 1)
            asteroid['dist_rank'] = rank

    def _handle_selection(self, movement):
        # 1:up, 2:down, 3:left, 4:right
        target_angles = {1: -math.pi / 2, 2: math.pi / 2, 3: math.pi, 4: 0}
        target_angle = target_angles[movement]
        
        available_asteroids = [(i, a) for i, a in enumerate(self.asteroids) if a['status'] == 'neutral']
        if not available_asteroids:
            self.selected_asteroid_idx = None
            return

        if self.selected_asteroid_idx is None or self.asteroids[self.selected_asteroid_idx]['status'] != 'neutral':
            current_pos = self.player_ship['pos']
        else:
            current_pos = self.asteroids[self.selected_asteroid_idx]['pos']

        best_candidate = -1
        min_score = float('inf')

        for i, asteroid in available_asteroids:
            if self.selected_asteroid_idx is not None and i == self.selected_asteroid_idx:
                continue
            
            vec = asteroid['pos'] - current_pos
            if np.linalg.norm(vec) < 1e-6: continue

            angle = math.atan2(vec[1], vec[0])
            angle_diff = (angle - target_angle + math.pi) % (2 * math.pi) - math.pi
            
            # Score prioritizes angle, with distance as a tie-breaker
            score = abs(angle_diff) * 100 + np.linalg.norm(vec)
            
            if score < min_score:
                min_score = score
                best_candidate = i
        
        if best_candidate != -1:
            self.selected_asteroid_idx = best_candidate
            # SFX: player.play_sound('select')

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            p['vel'][1] += 0.05 # Gravity

    def _create_capture_particles(self, pos):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': self.np_random.integers(20, 40),
                'color': random.choice([self.COLOR_PLAYER, (100, 220, 255), (200, 255, 255)])
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Stars
        for x, y, size in self.stars:
            pygame.draw.circle(self.screen, self.COLOR_STAR, (x, y), size * 0.5)

        # Last move trail
        if self.last_move_path:
            pygame.draw.line(self.screen, self.COLOR_PLAYER_TRAIL, self.last_move_path[0], self.last_move_path[1], 3)

        # Asteroids
        for i, asteroid in enumerate(self.asteroids):
            color = self.COLOR_ASTEROID_NEUTRAL
            if asteroid['status'] == 'captured':
                color = self.COLOR_ASTEROID_CAPTURED
            
            self._draw_iso_poly(asteroid['screen_pos'], asteroid['size'], color)

            if i == self.selected_asteroid_idx:
                self._draw_selection_indicator(asteroid['screen_pos'], asteroid['size'])

        # Player Ship
        self.player_ship['screen_pos'] = self._project(*self.player_ship['pos'])
        self._draw_iso_poly(self.player_ship['screen_pos'], 6, self.COLOR_PLAYER, is_ship=True)
        
        # Particles
        for p in self.particles:
            alpha = max(0, min(255, int(p['life'] * 6)))
            color = p['color'] + (alpha,)
            temp_surf = pygame.Surface((4, 4), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (2, 2), 2)
            self.screen.blit(temp_surf, (int(p['pos'][0]-2), int(p['pos'][1]-2)))

    def _draw_iso_poly(self, screen_pos, size, color, is_ship=False):
        x, y = screen_pos
        if is_ship:
            points = [
                (x, y - size * 1.5),
                (x - size, y),
                (x, y + size * 0.5),
                (x + size, y)
            ]
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
        else:
            points = [
                (x, y - size), (x + size * 1.5, y),
                (x, y + size), (x - size * 1.5, y)
            ]
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
            
            # 3D effect
            darker_color = tuple(c * 0.6 for c in color[:3])
            side_points = [
                (x, y + size), (x - size * 1.5, y),
                (x - size * 1.5, y + size), (x, y + 2 * size)
            ]
            pygame.gfxdraw.filled_polygon(self.screen, side_points, darker_color)

    def _draw_selection_indicator(self, screen_pos, size):
        x, y = screen_pos
        radius = int(size * 2.5)
        pulse = abs(math.sin(self.steps * 0.2))
        color = self.COLOR_ASTEROID_SELECTED
        
        # Pulsating brackets
        bracket_size = 8
        alpha = 150 + 105 * pulse
        temp_surf = pygame.Surface((radius*2 + 20, radius*2 + 20), pygame.SRCALPHA)
        
        # Top-left
        pygame.draw.lines(temp_surf, color + (alpha,), False, [(0, bracket_size), (0,0), (bracket_size, 0)], 2)
        # Top-right
        pygame.draw.lines(temp_surf, color + (alpha,), False, [(radius*2+20-bracket_size, 0), (radius*2+20, 0), (radius*2+20, bracket_size)], 2)
        # Bottom-left
        pygame.draw.lines(temp_surf, color + (alpha,), False, [(0, radius*2+20-bracket_size), (0, radius*2+20), (bracket_size, radius*2+20)], 2)
        # Bottom-right
        pygame.draw.lines(temp_surf, color + (alpha,), False, [(radius*2+20-bracket_size, radius*2+20), (radius*2+20, radius*2+20), (radius*2+20, radius*2+20-bracket_size)], 2)
        
        self.screen.blit(temp_surf, (x - radius - 10, y - radius - 10))


    def _render_ui(self):
        # UI Panel
        ui_panel = pygame.Surface((self.SCREEN_WIDTH, 40), pygame.SRCALPHA)
        ui_panel.fill(self.COLOR_UI_BG)
        self.screen.blit(ui_panel, (0, 0))
        
        # Text
        moves_text = self.font_small.render(f"MOVES: {self.moves_left}/{self.MAX_MOVES}", True, self.COLOR_TEXT)
        captured_text = self.font_small.render(f"CAPTURED: {self.captured_count}/{self.WIN_CONDITION_CAPTURES}", True, self.COLOR_TEXT)
        
        self.screen.blit(moves_text, (10, 10))
        self.screen.blit(captured_text, (self.SCREEN_WIDTH - captured_text.get_width() - 10, 10))
        
        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            if self.win:
                msg = "MISSION COMPLETE"
                color = self.COLOR_GAMEOVER_WIN
            else:
                msg = "MISSION FAILED"
                color = self.COLOR_GAMEOVER_LOSE
            
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "captured_count": self.captured_count,
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
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Asteroid Capture")
    clock = pygame.time.Clock()
    
    running = True
    terminated = False
    
    # Game loop for human play
    while running:
        action = [0, 0, 0] # [movement, space, shift]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                terminated = False

        if not terminated:
            keys = pygame.key.get_pressed()
            
            # Map keys to MultiDiscrete action space
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

            # Since auto_advance is False, we only step when there's an action
            # For human play, we want to step on every key press to see selection changes
            # So, we need a way to register a single press
            # A simple approach: step every frame with the current key state.
            # The env's logic prevents multiple moves on one key hold.
            obs, reward, terminated, truncated, info = env.step(action)
            
            if info.get('score', 0) != 0:
                print(f"Step: {info['steps']}, Reward: {reward:.2f}, Total Score: {info['score']:.2f}, Terminated: {terminated}")
        
        # Render the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) # Limit human play to 30 FPS
        
    env.close()