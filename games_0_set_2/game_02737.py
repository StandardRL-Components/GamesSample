
# Generated: 2025-08-28T05:47:25.930637
# Source Brief: brief_02737.md
# Brief Index: 2737

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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
        "Controls: Arrow keys to move. Press space to squash nearby bugs."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "An isometric arcade game. Squash as many bugs as you can before the time runs out!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30

        # Colors
        self.COLOR_BG = (20, 25, 30)
        self.COLOR_GRID = (40, 50, 60)
        self.COLOR_PLAYER = (0, 255, 128)
        self.COLOR_SHADOW = (10, 12, 15)
        self.COLOR_SPLAT = (200, 0, 0)
        self.BUG_COLORS = [(0, 200, 255), (255, 100, 200), (255, 255, 0)]
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_UI_TIMER_WARN = (255, 100, 100)

        # Game parameters
        self.TOTAL_GAME_TIME = 180  # seconds
        self.STAGE_DURATION = 60  # seconds
        self.MAX_STEPS = self.TOTAL_GAME_TIME * self.FPS
        self.WIN_SCORE = 50
        self.MAX_BUGS = 20
        self.INITIAL_SPAWN_RATE = 0.5  # bugs per second
        self.SPAWN_RATE_INCREASE = 0.25 # per stage

        # Player parameters
        self.PLAYER_SPEED = 3.0
        self.SQUASH_RADIUS = 30
        self.SQUASH_COOLDOWN = 10 # frames

        # Isometric projection parameters
        self.TILE_WIDTH = 32
        self.TILE_HEIGHT = 16
        self.ORIGIN_X = self.WIDTH // 2
        self.ORIGIN_Y = 80
        # World boundaries in cartesian coordinates
        self.WORLD_BOUNDS = pygame.Rect(0, 0, 22, 22)

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 24)
        
        # --- State Variables ---
        # These are initialized here but properly set in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = 0
        self.stage = 0
        self.current_spawn_rate = 0.0
        self.bug_spawn_accumulator = 0.0
        
        self.player_pos = np.zeros(2, dtype=np.float32)
        self.bugs = []
        self.particles = []
        
        self.prev_space_held = False
        self.squash_cooldown_timer = 0
        
        self.np_random = None

        # --- Validation ---
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = self.TOTAL_GAME_TIME
        self.stage = 1
        
        self.current_spawn_rate = self.INITIAL_SPAWN_RATE
        self.bug_spawn_accumulator = 0.0
        
        # Center player in the world
        self.player_pos = np.array([self.WORLD_BOUNDS.width / 2, self.WORLD_BOUNDS.height / 2], dtype=np.float32)
        
        self.bugs = []
        self.particles = []
        for _ in range(5): # Start with a few bugs
            self._spawn_bug()
            
        self.prev_space_held = False
        self.squash_cooldown_timer = 0
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        
        # --- Handle Input and Player Movement ---
        dist_before = self._get_dist_to_nearest_bug()
        self._handle_player_input(action)
        dist_after = self._get_dist_to_nearest_bug()

        if dist_after is not None and dist_before is not None:
            if dist_after < dist_before:
                reward += 0.1

        # --- Update Game State ---
        self._update_bugs()
        self._update_particles()
        self._update_spawner()
        self._update_timer_and_stage()
        
        # Handle squash action
        space_held = action[1] == 1
        if space_held and not self.prev_space_held and self.squash_cooldown_timer <= 0:
            # Sfx: Squash
            squashed_count = self._perform_squash()
            if squashed_count > 0:
                reward += squashed_count * 1.0
                self.score += squashed_count
            self.squash_cooldown_timer = self.SQUASH_COOLDOWN
        self.prev_space_held = space_held
        if self.squash_cooldown_timer > 0:
            self.squash_cooldown_timer -= 1

        # --- Check Termination ---
        self.steps += 1
        terminated = False
        if self.time_remaining <= 0:
            terminated = True
            reward = -10.0 # Penalty for timeout
        if self.score >= self.WIN_SCORE:
            terminated = True
            reward += 50.0 # Bonus for winning
            
        if terminated:
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    # --- Helper Functions for State Update ---

    def _handle_player_input(self, action):
        movement = action[0]
        move_vec = np.zeros(2, dtype=np.float32)
        
        # Isometric movement: up/down/left/right correspond to diagonal movement
        if movement == 1: # Up
            move_vec += [-1, -1]
        elif movement == 2: # Down
            move_vec += [1, 1]
        elif movement == 3: # Left
            move_vec += [-1, 1]
        elif movement == 4: # Right
            move_vec += [1, -1]
        
        if np.linalg.norm(move_vec) > 0:
            move_vec = move_vec / np.linalg.norm(move_vec)
            self.player_pos += move_vec * self.PLAYER_SPEED / self.FPS
        
        # Clamp player position to world bounds
        self.player_pos[0] = np.clip(self.player_pos[0], 0, self.WORLD_BOUNDS.width)
        self.player_pos[1] = np.clip(self.player_pos[1], 0, self.WORLD_BOUNDS.height)

    def _update_bugs(self):
        for bug in self.bugs:
            # Simple random walk logic
            if self.np_random.random() < 0.05: # 5% chance to change direction
                bug['target'] = self.np_random.uniform(
                    low=[0, 0], 
                    high=[self.WORLD_BOUNDS.width, self.WORLD_BOUNDS.height], 
                    size=2
                ).astype(np.float32)
            
            direction = bug['target'] - bug['pos']
            dist = np.linalg.norm(direction)
            if dist > 1.0:
                direction /= dist
                bug['pos'] += direction * bug['speed'] / self.FPS
                bug['pos'][0] = np.clip(bug['pos'][0], 0, self.WORLD_BOUNDS.width)
                bug['pos'][1] = np.clip(bug['pos'][1], 0, self.WORLD_BOUNDS.height)

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['life'] -= 1
            p['radius'] += p['growth']
            p['alpha'] = max(0, p['alpha'] - p['fade'])

    def _update_spawner(self):
        if len(self.bugs) < self.MAX_BUGS:
            self.bug_spawn_accumulator += self.current_spawn_rate / self.FPS
            if self.bug_spawn_accumulator >= 1.0:
                self._spawn_bug()
                self.bug_spawn_accumulator -= 1.0

    def _update_timer_and_stage(self):
        self.time_remaining -= 1 / self.FPS
        
        current_stage = math.floor((self.TOTAL_GAME_TIME - self.time_remaining) / self.STAGE_DURATION) + 1
        if current_stage > self.stage and current_stage <= 3:
            self.stage = current_stage
            self.current_spawn_rate = self.INITIAL_SPAWN_RATE + (self.stage - 1) * self.SPAWN_RATE_INCREASE
            # Sfx: Stage Up

    def _spawn_bug(self):
        # Ensure bug doesn't spawn on top of player
        while True:
            pos = self.np_random.uniform(
                low=[0, 0], 
                high=[self.WORLD_BOUNDS.width, self.WORLD_BOUNDS.height], 
                size=2
            ).astype(np.float32)
            if np.linalg.norm(pos - self.player_pos) > self.SQUASH_RADIUS / 10: # Use world distance
                break
        
        self.bugs.append({
            'pos': pos,
            'target': pos.copy(),
            'color': random.choice(self.BUG_COLORS),
            'size': self.np_random.integers(4, 7),
            'speed': self.np_random.uniform(1.0, 2.5),
        })

    def _perform_squash(self):
        squashed_bugs = []
        squashed_count = 0
        for bug in self.bugs:
            # Use screen-space distance for squash radius for intuitive feel
            dist = np.linalg.norm(self._cart_to_iso(bug['pos']) - self._cart_to_iso(self.player_pos))
            if dist < self.SQUASH_RADIUS:
                squashed_bugs.append(bug)
                squashed_count += 1
        
        for bug in squashed_bugs:
            self.bugs.remove(bug)
            # Create splat particle
            screen_pos = self._cart_to_iso(bug['pos'])
            self._create_splat(screen_pos, bug['color'])
            
        return squashed_count

    def _create_splat(self, pos, color):
        # Main splat
        self.particles.append({
            'pos': pos, 'radius': 5, 'life': 20, 'color': self.COLOR_SPLAT,
            'alpha': 200, 'fade': 10, 'growth': 0.5
        })
        # Smaller droplets
        for _ in range(self.np_random.integers(5, 10)):
            angle = self.np_random.uniform(0, 2 * math.pi)
            dist = self.np_random.uniform(5, 20)
            p_pos = (pos[0] + math.cos(angle) * dist, pos[1] + math.sin(angle) * dist)
            self.particles.append({
                'pos': p_pos, 'radius': self.np_random.uniform(1, 3), 'life': 15,
                'color': color, 'alpha': 255, 'fade': 15, 'growth': -0.1
            })

    def _get_dist_to_nearest_bug(self):
        if not self.bugs:
            return None
        bug_positions = np.array([bug['pos'] for bug in self.bugs])
        distances = np.linalg.norm(bug_positions - self.player_pos, axis=1)
        return np.min(distances)

    # --- Rendering and Observation ---

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._draw_grid()
        
        # Sort entities by Y position for correct isometric drawing
        entities = []
        entities.append({'type': 'player', 'pos': self.player_pos})
        for bug in self.bugs:
            entities.append({'type': 'bug', 'pos': bug['pos'], 'data': bug})
            
        entities.sort(key=lambda e: e['pos'][0] + e['pos'][1])
        
        for entity in entities:
            screen_pos = self._cart_to_iso(entity['pos'])
            shadow_pos = (screen_pos[0], screen_pos[1] + 12)
            
            if entity['type'] == 'player':
                pygame.gfxdraw.filled_ellipse(self.screen, int(shadow_pos[0]), int(shadow_pos[1]), 10, 5, self.COLOR_SHADOW)
                pygame.gfxdraw.filled_circle(self.screen, int(screen_pos[0]), int(screen_pos[1]), 8, self.COLOR_PLAYER)
                pygame.gfxdraw.aacircle(self.screen, int(screen_pos[0]), int(screen_pos[1]), 8, self.COLOR_PLAYER)
            elif entity['type'] == 'bug':
                bug_data = entity['data']
                pygame.gfxdraw.filled_ellipse(self.screen, int(shadow_pos[0]), int(shadow_pos[1]), bug_data['size'], int(bug_data['size'] * 0.5), self.COLOR_SHADOW)
                pygame.gfxdraw.filled_circle(self.screen, int(screen_pos[0]), int(screen_pos[1]), bug_data['size'], bug_data['color'])
                pygame.gfxdraw.aacircle(self.screen, int(screen_pos[0]), int(screen_pos[1]), bug_data['size'], bug_data['color'])

        self._render_particles()

    def _render_particles(self):
        for p in self.particles:
            if p['life'] > 0:
                radius = max(0, int(p['radius']))
                if radius > 0:
                    surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
                    pygame.draw.circle(surf, (*p['color'], int(p['alpha'])), (radius, radius), radius)
                    self.screen.blit(surf, (p['pos'][0] - radius, p['pos'][1] - radius))

    def _draw_grid(self):
        for i in range(self.WORLD_BOUNDS.width + 1):
            start_iso = self._cart_to_iso(np.array([i, 0]))
            end_iso = self._cart_to_iso(np.array([i, self.WORLD_BOUNDS.height]))
            pygame.draw.aaline(self.screen, self.COLOR_GRID, start_iso, end_iso)
        for i in range(self.WORLD_BOUNDS.height + 1):
            start_iso = self._cart_to_iso(np.array([0, i]))
            end_iso = self._cart_to_iso(np.array([self.WORLD_BOUNDS.width, i]))
            pygame.draw.aaline(self.screen, self.COLOR_GRID, start_iso, end_iso)

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"{self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, 10))
        
        # Timer
        time_str = f"{max(0, math.ceil(self.time_remaining))}"
        timer_color = self.COLOR_UI_TEXT if self.time_remaining > 10 else self.COLOR_UI_TIMER_WARN
        timer_text = self.font_large.render(time_str, True, timer_color)
        self.screen.blit(timer_text, (self.WIDTH - timer_text.get_width() - 20, 10))

        # Stage
        stage_text = self.font_medium.render(f"Stage {self.stage}", True, self.COLOR_UI_TEXT)
        stage_rect = stage_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT - 30))
        self.screen.blit(stage_text, stage_rect)

        # Game Over / Win message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            if self.score >= self.WIN_SCORE:
                msg = "YOU WIN!"
            else:
                msg = "TIME'S UP!"
            
            end_text = self.font_large.render(msg, True, (255, 255, 255))
            end_rect = end_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(end_text, end_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining,
            "stage": self.stage,
            "bugs_on_screen": len(self.bugs),
        }

    # --- Coordinate Conversion ---

    def _cart_to_iso(self, cart_pos):
        iso_x = self.ORIGIN_X + (cart_pos[0] - cart_pos[1]) * (self.TILE_WIDTH / 2)
        iso_y = self.ORIGIN_Y + (cart_pos[0] + cart_pos[1]) * (self.TILE_HEIGHT / 2)
        return np.array([iso_x, iso_y])

    def close(self):
        pygame.quit()

    # --- Validation ---

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

if __name__ == '__main__':
    # This block allows you to run the file directly to test the environment
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Use a display for human testing
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Bug Squasher")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        # Human input mapping
        keys = pygame.key.get_pressed()
        
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation from the environment to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            obs, info = env.reset()
            total_reward = 0
            
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        clock.tick(env.FPS)
        
    env.close()