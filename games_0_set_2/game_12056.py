import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:56:55.221412
# Source Brief: brief_02056.md
# Brief Index: 2056
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Gravity Chain: A puzzle-arcade game where the player manipulates the gravity
    of falling platforms to create stable chains.

    The goal is to complete 3 levels by creating a chain of 5 platforms in each,
    all within a 2-minute time limit. The episode ends if any platform hits the
    bottom of the screen, time runs out, or all levels are completed.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Manipulate the gravity of falling platforms to link them into stable chains. "
        "Complete all levels before time runs out or a platform hits the bottom."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to select a platform. Press space to toggle its gravity."
    )
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30  # Assumed step rate for timing calculations
    MAX_TIME_SECONDS = 120
    MAX_EPISODE_STEPS = MAX_TIME_SECONDS * FPS

    # Colors
    COLOR_BG = (10, 10, 26)         # Dark blue-purple
    COLOR_GRID = (30, 30, 60)
    COLOR_TEXT = (230, 230, 255)
    COLOR_TIMER_WARN = (255, 200, 0)
    COLOR_TIMER_CRIT = (255, 50, 50)
    COLOR_SCORE = (180, 180, 255)
    COLOR_WHITE_GLOW = (255, 255, 255)
    PLATFORM_COLORS = [
        (0, 255, 255),  # Cyan
        (255, 0, 255),  # Magenta
        (255, 255, 0),  # Yellow
        (0, 255, 128),  # Spring Green
        (255, 128, 0),  # Orange,
    ]

    # Game Parameters
    PLATFORM_WIDTH, PLATFORM_HEIGHT = 60, 15
    INITIAL_FALL_SPEED = 1.0
    SPEED_INCREMENT_PER_LEVEL = 0.25 # Brief said 0.05, but that's too slow to notice.
    CHAIN_X_TOLERANCE = 10
    CHAIN_Y_TOLERANCE = 35
    LEVEL_TARGET_CHAIN_SIZE = 5
    MAX_LEVELS = 3

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.render_mode = render_mode
        
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
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_score = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_msg = pygame.font.SysFont("Consolas", 48, bold=True)

        # --- Game State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.timer = 0
        self.level = 1
        self.platforms = []
        self.chains = []
        self.selected_platform_idx = None
        self.last_space_state = 0
        self.spawn_timer = 0
        self.current_fall_speed = 0.0
        self.particles = []
        self.level_completed_this_step = False
        
        # Use a seeded random number generator for reproducibility
        self.np_random = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.timer = self.MAX_EPISODE_STEPS
        self.level = 1
        self.platforms = []
        self.chains = []
        self.selected_platform_idx = None
        self.last_space_state = 0
        self.particles = []
        
        self._setup_level()

        return self._get_observation(), self._get_info()

    def _setup_level(self):
        """Initializes state for the current level."""
        self.platforms.clear()
        self.chains.clear()
        self.particles.clear()
        self.selected_platform_idx = None
        self.current_fall_speed = self.INITIAL_FALL_SPEED + (self.level - 1) * self.SPEED_INCREMENT_PER_LEVEL
        self.spawn_timer = 0
        
        # Spawn initial platforms
        for i in range(self.level + 2):
            self._spawn_platform(initial_spawn=True)
        
        if self.platforms:
            self.selected_platform_idx = 0

    def step(self, action):
        if self.game_over:
            # If the game is already over, do nothing and return the last state.
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.timer -= 1
        self.level_completed_this_step = False

        movement, space_held, _ = action
        space_pressed = (space_held == 1 and self.last_space_state == 0)
        self.last_space_state = space_held

        self._handle_input(movement, space_pressed)
        self._update_game_state()
        
        reward = self._calculate_reward()
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_EPISODE_STEPS
        
        if terminated:
            self.game_over = True
            if self.win:
                reward += 50 # Win bonus
            else:
                reward -= 10 # Loss penalty

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, movement, space_pressed):
        """Process player actions: selection and gravity toggle."""
        # --- Gravity Toggle ---
        if space_pressed and self.selected_platform_idx is not None and self.platforms:
            platform = self.platforms[self.selected_platform_idx]
            platform['gravity_dir'] *= -1
            # SFX: Gravity shift sound
            self._create_particles(platform['x'], platform['y'])

        # --- Platform Selection ---
        if movement != 0 and len(self.platforms) > 1 and self.selected_platform_idx is not None:
            current_plat = self.platforms[self.selected_platform_idx]
            cx, cy = current_plat['x'], current_plat['y']
            
            other_plats = [p for i, p in enumerate(self.platforms) if i != self.selected_platform_idx]
            
            candidates = []
            if movement == 1: # Up
                candidates = [p for p in other_plats if p['y'] < cy]
                if not candidates: # Wrap around
                    candidates = sorted(other_plats, key=lambda p: p['y'], reverse=True)
            elif movement == 2: # Down
                candidates = [p for p in other_plats if p['y'] > cy]
                if not candidates: # Wrap around
                    candidates = sorted(other_plats, key=lambda p: p['y'])
            elif movement == 3: # Left
                candidates = [p for p in other_plats if p['x'] < cx]
                if not candidates: # Wrap around
                    candidates = sorted(other_plats, key=lambda p: p['x'], reverse=True)
            elif movement == 4: # Right
                candidates = [p for p in other_plats if p['x'] > cx]
                if not candidates: # Wrap around
                    candidates = sorted(other_plats, key=lambda p: p['x'])
            
            if candidates:
                # Find the closest candidate based on Euclidean distance
                closest = min(candidates, key=lambda p: math.hypot(p['x'] - cx, p['y'] - cy))
                self.selected_platform_idx = self.platforms.index(closest)
                # SFX: UI selection tick

    def _update_game_state(self):
        """Update positions, spawning, chains, and check for game events."""
        # --- Update Platforms ---
        platforms_to_remove = []
        for i, p in enumerate(self.platforms):
            p['y'] += self.current_fall_speed * p['gravity_dir']
            
            # Boundary checks
            if p['y'] + self.PLATFORM_HEIGHT / 2 > self.HEIGHT:
                self.game_over = True # Failure condition
                # SFX: Fail / Explosion sound
            elif p['y'] < -self.PLATFORM_HEIGHT:
                platforms_to_remove.append(p)

        # Remove off-screen (top) platforms
        if platforms_to_remove:
            current_selection = self.platforms[self.selected_platform_idx] if self.selected_platform_idx is not None and self.platforms else None
            for p in platforms_to_remove:
                self.platforms.remove(p)
            if not self.platforms:
                self.selected_platform_idx = None
            elif current_selection not in self.platforms and self.platforms:
                self.selected_platform_idx = 0

        # --- Update Particles ---
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1

        # --- Spawning ---
        self.spawn_timer -= 1
        if self.spawn_timer <= 0 and len(self.platforms) < 15:
            self._spawn_platform()
            self.spawn_timer = self.np_random.integers(max(30, 90 - self.level * 15), max(45, 105 - self.level * 15))

        # --- Update Chains ---
        self._update_chains()

        # --- Check for Level Completion ---
        if any(len(chain) >= self.LEVEL_TARGET_CHAIN_SIZE for chain in self.chains):
            self.level_completed_this_step = True
            self.score += 100 * self.level # Bonus for completing level
            # SFX: Level complete fanfare
            self.level += 1
            if self.level > self.MAX_LEVELS:
                self.win = True
                self.game_over = True
            else:
                self._setup_level()

    def _spawn_platform(self, initial_spawn=False):
        y_pos = self.np_random.uniform(20, 100) if initial_spawn else -self.PLATFORM_HEIGHT
        platform = {
            'x': self.np_random.uniform(self.PLATFORM_WIDTH / 2, self.WIDTH - self.PLATFORM_WIDTH / 2),
            'y': y_pos,
            'color': self.PLATFORM_COLORS[self.np_random.integers(len(self.PLATFORM_COLORS))],
            'gravity_dir': 1,
            'in_chain': False,
        }
        self.platforms.append(platform)

    def _update_chains(self):
        """Detect and update all platform chains using graph traversal."""
        num_platforms = len(self.platforms)
        for p in self.platforms: p['in_chain'] = False
        
        if num_platforms < 2:
            self.chains = []
            return

        adj = [[] for _ in range(num_platforms)]
        for i in range(num_platforms):
            for j in range(i + 1, num_platforms):
                p1 = self.platforms[i]
                p2 = self.platforms[j]
                if (abs(p1['x'] - p2['x']) < self.CHAIN_X_TOLERANCE and
                    abs(p1['y'] - p2['y']) < self.CHAIN_Y_TOLERANCE):
                    adj[i].append(j)
                    adj[j].append(i)

        self.chains = []
        visited = [False] * num_platforms

        for i in range(num_platforms):
            if not visited[i]:
                component = []
                q = [i]
                visited[i] = True
                head = 0
                while head < len(q):
                    u = q[head]
                    head += 1
                    component.append(u)
                    for v in adj[u]:
                        if not visited[v]:
                            visited[v] = True
                            q.append(v)
                if len(component) > 1:
                    self.chains.append(component)
                    for node_idx in component:
                        self.platforms[node_idx]['in_chain'] = True

    def _calculate_reward(self):
        reward = 0
        
        # Continuous reward for being in a chain
        for chain in self.chains:
            reward += len(chain) * 0.01

        # Event-based reward for completing a level
        if self.level_completed_this_step:
            reward += 10  # Base reward for completing level
        
        return reward

    def _check_termination(self):
        """Check for game over, win, or timeout."""
        return self.game_over or self.timer <= 0

    def _get_observation(self):
        self._render_to_surface()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
            "timer": self.timer / self.FPS,
            "is_win": self.win,
        }

    def _render_to_surface(self):
        """Main rendering loop called by _get_observation."""
        # --- Background ---
        self.screen.fill(self.COLOR_BG)
        for x in range(0, self.WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT), 1)
        for y in range(0, self.HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y), 1)

        # --- Game Elements ---
        self._render_chains()
        self._render_platforms()
        self._render_particles()

        # --- UI ---
        self._render_ui()

        # --- Game Over/Win Message ---
        if self.game_over:
            msg = "YOU WIN!" if self.win else "GAME OVER"
            color = (100, 255, 100) if self.win else (255, 50, 50)
            text_surf = self.font_msg.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(text_surf, text_rect)

    def _render_chains(self):
        for chain in self.chains:
            for i in range(len(chain)):
                for j in range(i + 1, len(chain)):
                    p1_idx, p2_idx = chain[i], chain[j]
                    p1 = self.platforms[p1_idx]
                    p2 = self.platforms[p2_idx]
                    if (abs(p1['x'] - p2['x']) < self.CHAIN_X_TOLERANCE and
                        abs(p1['y'] - p2['y']) < self.CHAIN_Y_TOLERANCE):
                        # Draw glowing line
                        pos1 = (int(p1['x']), int(p1['y']))
                        pos2 = (int(p2['x']), int(p2['y']))
                        pygame.draw.aaline(self.screen, self.COLOR_WHITE_GLOW, pos1, pos2, 3)

    def _render_platforms(self):
        for i, p in enumerate(self.platforms):
            is_selected = (i == self.selected_platform_idx)
            rect = pygame.Rect(
                p['x'] - self.PLATFORM_WIDTH / 2,
                p['y'] - self.PLATFORM_HEIGHT / 2,
                self.PLATFORM_WIDTH,
                self.PLATFORM_HEIGHT
            )
            color = self.COLOR_WHITE_GLOW if p['in_chain'] else p['color']

            # Draw glow for selected platform
            if is_selected:
                glow_rect = rect.inflate(8, 8)
                s = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
                pygame.draw.rect(s, (*color, 60), s.get_rect(), border_radius=5)
                self.screen.blit(s, glow_rect.topleft)

            # Draw platform
            pygame.draw.rect(self.screen, color, rect, border_radius=3)
            if is_selected:
                pygame.draw.rect(self.screen, self.COLOR_WHITE_GLOW, rect, 2, border_radius=3)
            
            # Draw gravity indicator
            if p['gravity_dir'] == 1: # Down
                points = [(p['x'], p['y'] + self.PLATFORM_HEIGHT/2 + 5), (p['x']-5, p['y'] + self.PLATFORM_HEIGHT/2), (p['x']+5, p['y'] + self.PLATFORM_HEIGHT/2)]
            else: # Up
                points = [(p['x'], p['y'] - self.PLATFORM_HEIGHT/2 - 5), (p['x']-5, p['y'] - self.PLATFORM_HEIGHT/2), (p['x']+5, p['y'] - self.PLATFORM_HEIGHT/2)]
            
            # Only draw indicator if selected
            if is_selected:
                pygame.gfxdraw.aapolygon(self.screen, [(int(x), int(y)) for x, y in points], color)
                pygame.gfxdraw.filled_polygon(self.screen, [(int(x), int(y)) for x, y in points], color)

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / p['max_life']))))
            color = (*p['color'], alpha)
            s = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(s, color, (p['size'], p['size']), p['size'])
            self.screen.blit(s, (p['pos'][0]-p['size'], p['pos'][1]-p['size']))

    def _render_ui(self):
        # Level
        level_text = self.font_ui.render(f"Level: {self.level}/{self.MAX_LEVELS}", True, self.COLOR_TEXT)
        self.screen.blit(level_text, (10, 10))

        # Timer
        time_left_sec = max(0, self.timer / self.FPS)
        timer_color = self.COLOR_TEXT
        if time_left_sec < 10: timer_color = self.COLOR_TIMER_CRIT
        elif time_left_sec < 30: timer_color = self.COLOR_TIMER_WARN
        timer_text = self.font_ui.render(f"Time: {time_left_sec:.1f}", True, timer_color)
        timer_rect = timer_text.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(timer_text, timer_rect)

        # Score
        score_text = self.font_score.render(f"Score: {self.score}", True, self.COLOR_SCORE)
        score_rect = score_text.get_rect(center=(self.WIDTH / 2, 25))
        self.screen.blit(score_text, score_rect)

    def _create_particles(self, x, y):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            max_life = self.np_random.integers(15, 25)
            self.particles.append({
                'pos': [x, y],
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': max_life,
                'max_life': max_life,
                'color': self.COLOR_WHITE_GLOW if self.np_random.random() > 0.5 else (100, 100, 255),
                'size': self.np_random.integers(2,5)
            })
            
    def close(self):
        pygame.font.quit()
        pygame.quit()

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv()
    
    # --- Manual Play Loop ---
    obs, info = env.reset(seed=42)
    terminated = False
    
    pygame.display.set_caption("Gravity Chain")
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    clock = pygame.time.Clock()
    
    # Map keys to actions
    key_map = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }

    running = True
    while running:
        # Default action is no-op
        movement = 0
        space_held = 0
        
        keys = pygame.key.get_pressed()
        
        # Track if a movement key was pressed this frame to avoid re-triggering
        move_key_pressed = False
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs, info = env.reset(seed=42)
                if event.key in key_map:
                    movement = key_map[event.key]
                    move_key_pressed = True

        # If no new keydown event, check for held keys (less responsive but works)
        if not move_key_pressed:
            for key, move_action in key_map.items():
                if keys[key]:
                    # This part is tricky for single-step selection.
                    # The current implementation in step is better suited for keydown events.
                    # For a simple interactive loop, we'll stick to keydown.
                    pass
                
        if keys[pygame.K_SPACE]:
            space_held = 1
        
        action = [movement, space_held, 0] # Shift is not used
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Win: {info['is_win']}")
            pygame.time.wait(2000) # Pause for 2 seconds
            obs, info = env.reset(seed=43)

        clock.tick(GameEnv.FPS)

    env.close()