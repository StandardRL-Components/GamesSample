
# Generated: 2025-08-28T06:12:40.625719
# Source Brief: brief_05826.md
# Brief Index: 5826

        
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

    # User-facing control string
    user_guide = (
        "Controls: ↑↓←→ to move. Hold Shift or Space for a high-speed 'risky' dash."
    )

    # User-facing game description
    game_description = (
        "An isometric twist on classic Pong. Use risky high-speed moves to outmaneuver your opponent and score."
    )

    # Frames auto-advance at a fixed rate
    auto_advance = True

    # --- Constants ---
    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_COURT = (40, 50, 80)
    COLOR_LINES = (180, 180, 200)
    COLOR_PLAYER = (50, 150, 255)
    COLOR_PLAYER_RISKY = (150, 220, 255)
    COLOR_AI = (255, 100, 100)
    COLOR_BALL = (255, 255, 255)
    COLOR_TEXT = (240, 240, 240)
    COLOR_SCORE = (255, 220, 100)
    COLOR_PARTICLE = (200, 200, 255)
    
    # Screen Dimensions
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400

    # Isometric Projection
    ISO_SCALE = 24
    ISO_ANGLE = math.pi / 6

    # Game Settings
    WIN_SCORE = 5
    LOSE_SCORE = 3
    MAX_STEPS = 1800 # 60 seconds at 30fps

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Gymnasium Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame Setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 64)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 24)
        
        # Isometric transformation setup
        self.origin_x = self.SCREEN_WIDTH // 2
        self.origin_y = self.SCREEN_HEIGHT // 2 - 80

        # Court dimensions in isometric space
        self.court_width = 16
        self.court_height = 10
        
        # Initialize state variables
        self.steps = 0
        self.player_score = 0
        self.ai_score = 0
        self.game_over = False
        self.player_pos = None
        self.ai_pos = None
        self.ball_pos = None
        self.ball_vel = None
        self.ball_base_speed = 0.0
        self.player_risky_timer = 0
        self.last_hit_was_risky = False
        self.particles = []
        self.np_random = None

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.player_score = 0
        self.ai_score = 0
        self.game_over = False
        
        # Player paddle starts on the left side, centered
        self.player_pos = np.array([-self.court_width / 2 + 1.5, 0.0], dtype=np.float32)
        
        # AI paddle starts on the right side, centered
        self.ai_pos = np.array([self.court_width / 2 - 1.5, 0.0], dtype=np.float32)

        # Ball state
        self.ball_pos = np.array([0.0, 0.0], dtype=np.float32)
        self.ball_base_speed = 4.0 / 30.0 # pixels per frame, adjusted for 30fps
        
        # Start ball moving towards a random player
        angle = self.np_random.uniform(-math.pi / 4, math.pi / 4)
        direction = 1 if self.np_random.random() < 0.5 else -1
        self.ball_vel = np.array([math.cos(angle) * direction, math.sin(angle)], dtype=np.float32) * self.ball_base_speed

        self.player_risky_timer = 0
        self.last_hit_was_risky = False
        self.particles.clear()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = -0.01 # Small penalty for time passing
        
        # --- Handle Actions ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        is_risky_action = space_held or shift_held
        
        if self.player_risky_timer > 0:
            self.player_risky_timer -= 1
            paddle_speed = 6.0 / 30.0
        else:
            paddle_speed = 2.0 / 30.0

        if is_risky_action and self.player_risky_timer == 0:
            self.player_risky_timer = 10 # 1/3 of a second duration
            paddle_speed = 6.0 / 30.0
            self.last_hit_was_risky = True # Flag that a risky move was initiated
            # SFX: Risky dash sound

        # Update player paddle position
        if movement == 1: self.player_pos[1] -= paddle_speed # Up
        elif movement == 2: self.player_pos[1] += paddle_speed # Down
        elif movement == 3: self.player_pos[0] -= paddle_speed # Left
        elif movement == 4: self.player_pos[0] += paddle_speed # Right

        # Clamp player paddle to their side of the court
        self.player_pos[0] = np.clip(self.player_pos[0], -self.court_width / 2, -0.5)
        self.player_pos[1] = np.clip(self.player_pos[1], -self.court_height / 2, self.court_height / 2)

        # --- AI Logic ---
        ai_speed = 3.0 / 30.0
        target_y = self.ball_pos[1]
        if self.ai_pos[1] < target_y:
            self.ai_pos[1] += min(ai_speed, target_y - self.ai_pos[1])
        elif self.ai_pos[1] > target_y:
            self.ai_pos[1] -= min(ai_speed, self.ai_pos[1] - target_y)
        self.ai_pos[1] = np.clip(self.ai_pos[1], -self.court_height / 2, self.court_height / 2)

        # --- Ball Logic ---
        self.ball_pos += self.ball_vel
        
        # Ball collision with top/bottom walls
        if abs(self.ball_pos[1]) > self.court_height / 2:
            self.ball_vel[1] *= -1
            self.ball_pos[1] = np.clip(self.ball_pos[1], -self.court_height / 2, self.court_height / 2)
            self._create_particles(self.ball_pos, 5)
            # SFX: Wall bounce

        # Ball collision with paddles
        paddle_height = 3.0
        paddle_width = 0.5
        
        # Player paddle collision
        if (self.player_pos[0] - paddle_width < self.ball_pos[0] < self.player_pos[0] + paddle_width and
            self.player_pos[1] - paddle_height / 2 < self.ball_pos[1] < self.player_pos[1] + paddle_height / 2 and
            self.ball_vel[0] < 0):
            
            self.ball_vel[0] *= -1
            
            # Add deflection based on impact point
            offset = (self.ball_pos[1] - self.player_pos[1]) / (paddle_height / 2)
            self.ball_vel[1] += offset * 0.5 * self.ball_base_speed
            
            # Normalize velocity to maintain speed
            self.ball_vel = self.ball_vel / np.linalg.norm(self.ball_vel) * self.ball_base_speed
            
            reward += 0.1
            self._create_particles(self.ball_pos, 15, self.COLOR_PLAYER)
            # SFX: Player paddle hit

            if self.last_hit_was_risky:
                reward += 0.5
                # SFX: Risky hit success
            self.last_hit_was_risky = False # Reset flag after a successful hit

        # AI paddle collision
        if (self.ai_pos[0] - paddle_width < self.ball_pos[0] < self.ai_pos[0] + paddle_width and
            self.ai_pos[1] - paddle_height / 2 < self.ball_pos[1] < self.ai_pos[1] + paddle_height / 2 and
            self.ball_vel[0] > 0):

            self.ball_vel[0] *= -1
            offset = (self.ball_pos[1] - self.ai_pos[1]) / (paddle_height / 2)
            self.ball_vel[1] += offset * 0.5 * self.ball_base_speed
            self.ball_vel = self.ball_vel / np.linalg.norm(self.ball_vel) * self.ball_base_speed
            self._create_particles(self.ball_pos, 15, self.COLOR_AI)
            # SFX: AI paddle hit

        # --- Scoring Logic ---
        if self.ball_pos[0] > self.court_width / 2:
            self.player_score += 1
            reward += 1
            self.ball_base_speed += (0.2 / 30.0)
            self._reset_round(-1) # Ball moves towards AI
            # SFX: Player scores point
        elif self.ball_pos[0] < -self.court_width / 2:
            self.ai_score += 1
            reward -= 1
            if self.last_hit_was_risky:
                reward -= 0.2 # Penalty for failing after a risky move
            self.ball_base_speed += (0.2 / 30.0)
            self._reset_round(1) # Ball moves towards player
            # SFX: AI scores point

        # --- Particle Update ---
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1

        # --- Termination ---
        self.steps += 1
        terminated = False
        if self.player_score >= self.WIN_SCORE:
            reward += 10
            terminated = True
            self.game_over = True
        elif self.ai_score >= self.LOSE_SCORE:
            reward -= 10
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _reset_round(self, direction):
        self.ball_pos = np.array([0.0, 0.0], dtype=np.float32)
        angle = self.np_random.uniform(-math.pi / 6, math.pi / 6)
        self.ball_vel = np.array([math.cos(angle) * direction, math.sin(angle)], dtype=np.float32) * self.ball_base_speed
        self.player_risky_timer = 0
        self.last_hit_was_risky = False

    def _iso_to_screen(self, iso_pos):
        x = self.origin_x + (iso_pos[0] - iso_pos[1]) * self.ISO_SCALE * math.cos(self.ISO_ANGLE)
        y = self.origin_y + (iso_pos[0] + iso_pos[1]) * self.ISO_SCALE * math.sin(self.ISO_ANGLE)
        return int(x), int(y)

    def _draw_iso_rect(self, surface, color, iso_pos, width, height, border_width=0):
        # iso_pos is the center of the rectangle
        hw, hh = width / 2, height / 2
        points = [
            self._iso_to_screen((iso_pos[0] - hw, iso_pos[1] - hh)),
            self._iso_to_screen((iso_pos[0] + hw, iso_pos[1] - hh)),
            self._iso_to_screen((iso_pos[0] + hw, iso_pos[1] + hh)),
            self._iso_to_screen((iso_pos[0] - hw, iso_pos[1] + hh)),
        ]
        pygame.gfxdraw.aapolygon(surface, points, color)
        pygame.gfxdraw.filled_polygon(surface, points, color)
        if border_width > 0:
            pygame.draw.aalines(surface, self.COLOR_LINES, True, points, border_width)

    def _create_particles(self, iso_pos, count, color=None):
        screen_pos = self._iso_to_screen(iso_pos)
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            self.particles.append({
                'pos': np.array(screen_pos, dtype=np.float32),
                'vel': vel,
                'life': self.np_random.integers(10, 20),
                'color': color or self.COLOR_PARTICLE
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw court
        self._draw_iso_rect(self.screen, self.COLOR_COURT, (0, 0), self.court_width, self.court_height)
        
        # Draw court lines
        p1 = self._iso_to_screen((-self.court_width / 2, -self.court_height / 2))
        p2 = self._iso_to_screen((self.court_width / 2, -self.court_height / 2))
        p3 = self._iso_to_screen((self.court_width / 2, self.court_height / 2))
        p4 = self._iso_to_screen((-self.court_width / 2, self.court_height / 2))
        pygame.draw.aalines(self.screen, self.COLOR_LINES, True, [p1, p2, p3, p4])
        
        # Center line
        c1 = self._iso_to_screen((0, -self.court_height / 2))
        c2 = self._iso_to_screen((0, self.court_height / 2))
        pygame.draw.aaline(self.screen, self.COLOR_LINES, c1, c2)

        # Draw particles
        for p in self.particles:
            pygame.gfxdraw.pixel(self.screen, int(p['pos'][0]), int(p['pos'][1]), p['color'])

        # --- Draw elements ordered by depth (iso_y) ---
        entities = []
        entities.append({'type': 'paddle', 'pos': self.player_pos, 'color': self.COLOR_PLAYER if self.player_risky_timer == 0 else self.COLOR_PLAYER_RISKY, 'size': (0.5, 3.0)})
        entities.append({'type': 'paddle', 'pos': self.ai_pos, 'color': self.COLOR_AI, 'size': (0.5, 3.0)})
        entities.append({'type': 'ball', 'pos': self.ball_pos, 'color': self.COLOR_BALL, 'size': (0.7, 0.7)})
        
        entities.sort(key=lambda e: e['pos'][1])

        for entity in entities:
            if entity['type'] == 'paddle':
                self._draw_iso_rect(self.screen, entity['color'], entity['pos'], entity['size'][0], entity['size'][1])
            elif entity['type'] == 'ball':
                self._draw_iso_rect(self.screen, entity['color'], entity['pos'], entity['size'][0], entity['size'][1])

    def _render_ui(self):
        # Render scores
        player_score_surf = self.font_large.render(str(self.player_score), True, self.COLOR_SCORE)
        ai_score_surf = self.font_large.render(str(self.ai_score), True, self.COLOR_SCORE)
        
        self.screen.blit(player_score_surf, (self.SCREEN_WIDTH // 4 - player_score_surf.get_width() // 2, 20))
        self.screen.blit(ai_score_surf, (self.SCREEN_WIDTH * 3 // 4 - ai_score_surf.get_width() // 2, 20))

        # Render misses/lives for AI
        misses_text = "MISSES"
        misses_surf = self.font_small.render(misses_text, True, self.COLOR_TEXT)
        self.screen.blit(misses_surf, (self.SCREEN_WIDTH * 3 // 4 - misses_surf.get_width() // 2, 80))

        # Game Over Text
        if self.game_over:
            if self.player_score >= self.WIN_SCORE:
                text = "YOU WIN!"
            elif self.ai_score >= self.LOSE_SCORE:
                text = "GAME OVER"
            else: # Max steps reached
                text = "TIME'S UP"
            
            end_surf = self.font_large.render(text, True, self.COLOR_TEXT)
            end_rect = end_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_surf, end_rect)
    
    def _get_info(self):
        return {
            "score": self.player_score,
            "steps": self.steps,
            "ai_score": self.ai_score,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # --- Human Player Controls ---
    # Maps pygame keys to the MultiDiscrete action components
    key_map = {
        pygame.K_UP:    (1, 0, 0),
        pygame.K_DOWN:  (2, 0, 0),
        pygame.K_LEFT:  (3, 0, 0),
        pygame.K_RIGHT: (4, 0, 0),
    }

    # Pygame setup for human play
    pygame.display.set_caption("Isometric Pong")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    running = True
    
    total_reward = 0.0

    while running:
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0.0
                print("--- Game Reset ---")

        # --- Action Generation ---
        # Start with a no-op action
        action = np.array([0, 0, 0]) 
        
        keys = pygame.key.get_pressed()

        # Movement (non-exclusive, but we'll just take the first one found)
        for key, (move_val, _, _) in key_map.items():
            if keys[key]:
                action[0] = move_val
                break # Prioritize up/down/left/right in order of key_map
        
        # Space and Shift
        if keys[pygame.K_SPACE]:
            action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1

        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"Episode Finished. Final Score: {info['score']}-{info['ai_score']}. Total Reward: {total_reward:.2f}")
            obs, info = env.reset()
            total_reward = 0.0
            # Optional: Pause on game over until 'R' is pressed
            # running = False 

        # --- Rendering ---
        # Convert the observation (which is already a rendered frame) for display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) # Run at 30 FPS

    env.close()