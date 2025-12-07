import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:53:07.723590
# Source Brief: brief_01255.md
# Brief Index: 1255
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Jump between collapsing platforms in time with the beat. Land on-beat jumps to build your combo and score to survive as long as possible."
    )
    user_guide = (
        "Controls: Use the arrow keys (↑↓←→) to jump between platforms. Time your jumps with the beat to score points and build your combo."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 10, 7
    PLATFORM_SIZE = 50
    PLATFORM_GAP = 12
    PLATFORM_RADIUS = 10
    MAX_STEPS = 1800  # 60 seconds at 30 FPS

    # --- Colors ---
    COLOR_BG_TOP = (10, 5, 30)
    COLOR_BG_BOTTOM = (0, 0, 0)
    COLOR_PLAYER = (0, 191, 255)
    COLOR_PLAYER_GLOW = (0, 191, 255, 50)
    COLOR_PLATFORM_SAFE = (0, 255, 127)
    COLOR_PLATFORM_FALLING = (255, 69, 0)
    COLOR_PLATFORM_BORDER = (200, 200, 200)
    COLOR_BEAT_INDICATOR = (255, 255, 255)
    COLOR_TEXT = (220, 220, 220)
    COLOR_SCORE = (255, 215, 0)
    COLOR_COMBO = (0, 255, 255)
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        self.font_feedback = pygame.font.Font(None, 28)
        self.font_gameover = pygame.font.Font(None, 72)
        
        self.render_mode = render_mode

        # --- Dynamic Game Parameters ---
        self.initial_collapse_interval = 90 # 3 seconds at 30 FPS
        self.jump_duration = 10 # frames
        self.beat_interval = 45 # 1.5 seconds at 30 FPS
        self.beat_window = 4 # frames on each side of the beat

        self._initialize_state()

    def _initialize_state(self):
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.win = False

        self.grid = self._create_grid()
        
        self.player_grid_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.player_pixel_pos = self._grid_to_pixel(self.player_grid_pos)
        
        self.player_state = 'idle' # 'idle', 'jumping'
        self.jump_data = {
            'active': False,
            'start_pos': None,
            'target_grid_pos': None,
            'target_pixel_pos': None,
            'progress': 0.0
        }

        self.beat_timer = 0
        self.combo = 0
        self.combo_multiplier = 1.0

        self.collapse_timer = 0
        self.current_collapse_interval = self.initial_collapse_interval

        self.particles = []
        self.feedback_texts = []
        self.background_stars = [(random.randint(0, self.SCREEN_WIDTH), random.randint(0, self.SCREEN_HEIGHT), random.uniform(0.5, 1.5)) for _ in range(100)]


    def _create_grid(self):
        grid = []
        for y in range(self.GRID_HEIGHT):
            row = []
            for x in range(self.GRID_WIDTH):
                row.append({
                    'state': 'safe', # 'safe', 'falling', 'gone'
                    'fall_timer': -1,
                    'shake_offset': (0, 0)
                })
            grid.append(row)
        return grid

    def _grid_to_pixel(self, grid_pos):
        x = grid_pos[0] * (self.PLATFORM_SIZE + self.PLATFORM_GAP)
        y = grid_pos[1] * (self.PLATFORM_SIZE + self.PLATFORM_GAP)
        return [x, y]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._initialize_state()
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0.0

        self._update_timers_and_difficulty()
        
        reward += self._handle_player_action(action)
        reward += self._update_player_jump()
        
        self._update_platforms()
        self._update_grid_collapse()
        self._update_particles()
        self._update_feedback_text()

        terminated, terminal_reward = self._check_termination()
        reward += terminal_reward
        
        truncated = self.steps >= self.MAX_STEPS
        if truncated and not terminated: # Win condition
            terminated = True
            self.win = True
            terminal_reward = 50.0
            self._add_feedback_text("SURVIVED!", self.COLOR_SCORE, font_size=72)
            reward += terminal_reward
            self.game_over = True
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _update_timers_and_difficulty(self):
        self.beat_timer = (self.beat_timer + 1) % self.beat_interval
        self.collapse_timer += 1
        
        # Difficulty scaling: collapse interval decreases every 10 seconds (300 steps)
        if self.steps > 0 and self.steps % 300 == 0:
            self.current_collapse_interval = max(20, self.current_collapse_interval - 3) # Min interval of 2/3 second

    def _handle_player_action(self, action):
        movement = action[0]
        reward = 0.0

        if self.player_state == 'idle' and movement != 0:
            # --- JUMP INITIATION ---
            direction_map = {1: [0, -1], 2: [0, 1], 3: [-1, 0], 4: [1, 0]} # up, down, left, right
            direction = direction_map[movement]
            target_grid_pos = [self.player_grid_pos[0] + direction[0], self.player_grid_pos[1] + direction[1]]

            # Check if target is valid
            is_valid_target = (0 <= target_grid_pos[0] < self.GRID_WIDTH and
                               0 <= target_grid_pos[1] < self.GRID_HEIGHT and
                               self.grid[target_grid_pos[1]][target_grid_pos[0]]['state'] != 'gone')
            
            # Check if on beat
            is_on_beat = self.beat_timer < self.beat_window or self.beat_timer > self.beat_interval - self.beat_window

            if is_valid_target and is_on_beat:
                # SFX: Jump_Success
                self.player_state = 'jumping'
                self.jump_data['active'] = True
                self.jump_data['start_pos'] = self.player_pixel_pos[:]
                self.jump_data['target_grid_pos'] = target_grid_pos
                self.jump_data['target_pixel_pos'] = self._grid_to_pixel(target_grid_pos)
                self.jump_data['progress'] = 0.0
                
                self.combo += 1
                self.combo_multiplier = min(2.0, 1.0 + (self.combo -1) * 0.1)
                
                reward += 1.0 # On-beat initiation bonus
                self._create_particles(self._grid_to_pixel(self.player_grid_pos), self.COLOR_PLAYER, 20, 3, 5)
            else:
                # SFX: Jump_Fail
                if self.combo > 0:
                    self._add_feedback_text("Combo Lost!", self.COLOR_PLATFORM_FALLING)
                self.combo = 0
                self.combo_multiplier = 1.0
                reward -= 0.1 # Penalty for mistimed/invalid jump
        return reward

    def _update_player_jump(self):
        landing_reward = 0.0
        if self.player_state == 'jumping':
            self.jump_data['progress'] += 1.0 / self.jump_duration
            
            # Interpolate position with an arc
            progress = self.jump_data['progress']
            start = self.jump_data['start_pos']
            end = self.jump_data['target_pixel_pos']
            
            self.player_pixel_pos[0] = start[0] + (end[0] - start[0]) * progress
            self.player_pixel_pos[1] = start[1] + (end[1] - start[1]) * progress
            arc_height = math.sin(progress * math.pi) * 30
            self.player_pixel_pos[1] -= arc_height

            if self.jump_data['progress'] >= 1.0:
                # --- JUMP LANDING ---
                # SFX: Land_Success
                self.player_state = 'idle'
                self.jump_data['active'] = False
                self.player_grid_pos = self.jump_data['target_grid_pos']
                self.player_pixel_pos = self._grid_to_pixel(self.player_grid_pos)

                landed_platform = self.grid[self.player_grid_pos[1]][self.player_grid_pos[0]]
                
                base_reward = 5.0
                if landed_platform['state'] == 'falling':
                    base_reward += 2.0 # Risky landing bonus
                    self._add_feedback_text("Risky!", self.COLOR_PLAYER)
                else:
                    base_reward += 1.0 # Safe landing bonus
                
                total_jump_reward = base_reward * self.combo_multiplier
                landing_reward += total_jump_reward
                self.score += total_jump_reward
                self._add_feedback_text(f"+{total_jump_reward:.1f}", self.COLOR_SCORE)

                self._create_particles(self.player_pixel_pos, self.COLOR_PLATFORM_SAFE, 30, 5, 8)
        return landing_reward

    def _update_platforms(self):
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                platform = self.grid[y][x]
                if platform['state'] == 'falling':
                    platform['fall_timer'] -= 1
                    platform['shake_offset'] = (random.uniform(-2, 2), random.uniform(-2, 2))
                    if platform['fall_timer'] <= 0:
                        # SFX: Platform_Crumble
                        platform['state'] = 'gone'
                        self._create_particles(self._grid_to_pixel([x,y]), self.COLOR_PLATFORM_FALLING, 50, 2, 4, -2)

    def _update_grid_collapse(self):
        if self.collapse_timer >= self.current_collapse_interval:
            self.collapse_timer = 0
            safe_platforms = []
            for y in range(self.GRID_HEIGHT):
                for x in range(self.GRID_WIDTH):
                    if self.grid[y][x]['state'] == 'safe':
                        safe_platforms.append((x, y))
            
            if not safe_platforms: return

            random.shuffle(safe_platforms)

            for x, y in safe_platforms:
                # Anti-softlock: ensure player has at least one valid jump after collapse
                if self._is_collapse_survivable(x, y):
                    self.grid[y][x]['state'] = 'falling'
                    self.grid[y][x]['fall_timer'] = 60 # 2 seconds at 30 FPS
                    break

    def _is_collapse_survivable(self, collapse_x, collapse_y):
        px, py = self.player_grid_pos
        if px == collapse_x and py == collapse_y: # Player is on the collapsing platform
            return True # Can always jump off

        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx, ny = px + dx, py + dy
            if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT:
                if not (nx == collapse_x and ny == collapse_y):
                    if self.grid[ny][nx]['state'] != 'gone':
                        return True # Found a safe jump target
        return False # No safe jump targets

    def _check_termination(self):
        terminated = False
        terminal_reward = 0.0

        player_platform = self.grid[self.player_grid_pos[1]][self.player_grid_pos[0]]
        if self.player_state == 'idle' and player_platform['state'] == 'gone':
            # SFX: Player_Fall
            terminated = True
            terminal_reward = -100.0
            self._add_feedback_text("GAME OVER", self.COLOR_PLATFORM_FALLING, font_size=72)
        
        if terminated:
            self.game_over = True
        
        return terminated, terminal_reward

    def _get_observation(self):
        self._render_frame()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "combo": self.combo}

    def _render_frame(self):
        self._render_background()
        
        # Camera centers on player
        cam_offset_x = self.SCREEN_WIDTH / 2 - self.player_pixel_pos[0] - self.PLATFORM_SIZE / 2
        cam_offset_y = self.SCREEN_HEIGHT / 2 - self.player_pixel_pos[1] - self.PLATFORM_SIZE / 2

        self._render_platforms(cam_offset_x, cam_offset_y)
        self._render_particles(cam_offset_x, cam_offset_y)
        self._render_player() # Player is always rendered at the center
        self._render_beat_indicator()
        self._render_ui()

    def _render_background(self):
        self.screen.fill(self.COLOR_BG_BOTTOM)
        # Gradient
        for y in range(self.SCREEN_HEIGHT // 2):
            alpha = int(255 * (1 - (y / (self.SCREEN_HEIGHT // 2))))
            grad_color = self.COLOR_BG_TOP
            s = pygame.Surface((self.SCREEN_WIDTH, 1))
            s.set_alpha(alpha)
            s.fill(grad_color)
            self.screen.blit(s, (0, y))
        
        # Stars
        for star in self.background_stars:
            x, y, speed = star
            y_pos = (y + self.steps * speed * 0.1) % self.SCREEN_HEIGHT
            pygame.draw.circle(self.screen, (255,255,255), (int(x), int(y_pos)), 1)


    def _render_platforms(self, ox, oy):
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                platform = self.grid[y][x]
                if platform['state'] == 'gone':
                    continue
                
                px, py = self._grid_to_pixel([x, y])
                rect = pygame.Rect(
                    int(px + ox + platform['shake_offset'][0]),
                    int(py + oy + platform['shake_offset'][1]),
                    self.PLATFORM_SIZE,
                    self.PLATFORM_SIZE
                )
                
                color = self.COLOR_PLATFORM_SAFE if platform['state'] == 'safe' else self.COLOR_PLATFORM_FALLING
                if platform['state'] == 'falling':
                    alpha = 255 * (0.5 + 0.5 * math.sin(self.steps * 0.5))
                    color = (color[0], color[1], color[2], int(alpha))
                
                pygame.draw.rect(self.screen, color, rect, border_radius=self.PLATFORM_RADIUS)
                pygame.draw.rect(self.screen, self.COLOR_PLATFORM_BORDER, rect, 1, border_radius=self.PLATFORM_RADIUS)

    def _render_player(self):
        center_x, center_y = self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2
        
        # Glow effect
        glow_radius = int(12 + 4 * math.sin(self.steps * 0.1))
        s = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(s, self.COLOR_PLAYER_GLOW, (glow_radius, glow_radius), glow_radius)
        self.screen.blit(s, (center_x - glow_radius, center_y - glow_radius))

        # Player circle
        pygame.gfxdraw.aacircle(self.screen, center_x, center_y, 10, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, 10, self.COLOR_PLAYER)

    def _render_beat_indicator(self):
        center_x, center_y = self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2
        progress = self.beat_timer / self.beat_interval
        
        # Pulsing ring
        radius = int(20 + 40 * progress)
        alpha = int(255 * (1 - progress)**2)
        if alpha > 10:
            pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, (*self.COLOR_BEAT_INDICATOR, alpha))
        
        # On-beat flash
        is_on_beat = self.beat_timer < self.beat_window or self.beat_timer > self.beat_interval - self.beat_window
        if is_on_beat:
            flash_alpha = int(100 * (1 - abs(self.beat_timer if self.beat_timer < self.beat_interval/2 else self.beat_timer - self.beat_interval) / self.beat_window))
            s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            s.fill((255, 255, 255, flash_alpha))
            self.screen.blit(s, (0,0))


    def _render_ui(self):
        # Score
        score_surf = self.font_main.render(f"SCORE: {int(self.score)}", True, self.COLOR_SCORE)
        self.screen.blit(score_surf, (15, 10))

        # Combo
        if self.combo > 1:
            combo_surf = self.font_small.render(f"COMBO: x{self.combo_multiplier:.1f}", True, self.COLOR_COMBO)
            self.screen.blit(combo_surf, (15, 45))

        # Feedback Text
        for text_info in self.feedback_texts:
            font = self.font_gameover if text_info['font_size'] == 72 else self.font_feedback
            text_surf = font.render(text_info['text'], True, text_info['color'])
            text_surf.set_alpha(text_info['alpha'])
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, text_info['y']))
            self.screen.blit(text_surf, text_rect)

    def _create_particles(self, pos, color, count, min_speed, max_speed, y_offset=0):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(min_speed, max_speed)
            velocity = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                'pos': [pos[0] + self.PLATFORM_SIZE/2, pos[1] + self.PLATFORM_SIZE/2 + y_offset],
                'vel': velocity,
                'life': random.randint(15, 30),
                'color': color,
                'radius': random.uniform(2, 5)
            })
    
    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _render_particles(self, ox, oy):
        for p in self.particles:
            alpha = int(255 * (p['life'] / 30.0))
            color = (*p['color'], alpha)
            pos = (int(p['pos'][0] + ox), int(p['pos'][1] + oy))
            s = pygame.Surface((p['radius']*2, p['radius']*2), pygame.SRCALPHA)
            pygame.draw.circle(s, color, (p['radius'], p['radius']), p['radius'])
            self.screen.blit(s, (pos[0] - p['radius'], pos[1] - p['radius']))

    def _add_feedback_text(self, text, color, font_size=28):
        self.feedback_texts.append({
            'text': text,
            'color': color,
            'y': self.SCREEN_HEIGHT / 2 + 50,
            'alpha': 255,
            'life': 60,
            'font_size': font_size
        })

    def _update_feedback_text(self):
        for ft in self.feedback_texts[:]:
            ft['y'] -= 1
            ft['life'] -= 1
            ft['alpha'] = int(255 * (ft['life'] / 60.0))
            if ft['life'] <= 0:
                self.feedback_texts.remove(ft)

if __name__ == '__main__':
    # This block is for human play and debugging
    # It will not be run by the evaluation environment
    # but is helpful for development.
    
    # Un-comment the line below to run with a visible display
    os.environ.setdefault("SDL_VIDEODRIVER", "x11") 
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    done = False
    
    pygame.display.set_caption("Rhythm Grid Jumper")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    running = True
    total_reward = 0

    while running:
        movement = 0 # no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
                    done = False
                # Manual single-step for debugging
                if event.key == pygame.K_p:
                    action = [0,0,0] # no-op
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    print(f"Step: {info['steps']}, Reward: {reward}, Done: {done}, Info: {info}")


        if not done:
            # For continuous play, check pressed keys
            keys = pygame.key.get_pressed()
            # This logic is simplified for human play; an agent would learn the timing.
            # We check for a key press event to trigger a single jump action
            # instead of continuous holding.
            jump_triggered_this_frame = False
            for event in pygame.event.get(pygame.KEYDOWN):
                 if event.key in [pygame.K_UP, pygame.K_w]: movement = 1; jump_triggered_this_frame=True
                 elif event.key in [pygame.K_DOWN, pygame.K_s]: movement = 2; jump_triggered_this_frame=True
                 elif event.key in [pygame.K_LEFT, pygame.K_a]: movement = 3; jump_triggered_this_frame=True
                 elif event.key in [pygame.K_RIGHT, pygame.K_d]: movement = 4; jump_triggered_this_frame=True
            
            # The action is only the movement for this game. space/shift are unused.
            action = [movement, 0, 0]
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        # Render the observation to the display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(30) # Cap FPS for human play

    pygame.quit()