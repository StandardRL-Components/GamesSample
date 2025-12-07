import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T16:11:09.716174
# Source Brief: brief_00854.md
# Brief Index: 854
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A rhythm-based endless runner Gymnasium environment.

    The player runs through a procedurally generated abstract landscape.
    They must choose between musical portals appearing in four lanes to maintain
    their speed, which is synchronized with an increasing tempo.

    **Action Space:** MultiDiscrete([5, 2, 2])
    - `action[0]` (Movement):
        - 0: No action.
        - 1: Select lane 0 (top).
        - 2: Select lane 1.
        - 3: Select lane 2.
        - 4: Select lane 3 (bottom).
    - `action[1]` (Space): Unused.
    - `action[2]` (Shift): Unused.

    **Observation Space:** Box(0, 255, (400, 640, 3), uint8)
    - An RGB image of the game screen.

    **Rewards:**
    - Continuous reward for maintaining speed relative to the tempo.
    - Event-based reward for successfully traversing a portal.
    - Goal-oriented reward for distance milestones.
    - Terminal penalties/rewards for failure or completion.
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    game_description = (
        "A rhythm-based endless runner. Select the correct lane to hit musical portals on the beat, "
        "maintaining your speed through a procedurally generated abstract world."
    )
    user_guide = (
        "Controls: Use actions (e.g., arrow keys) to select one of the four lanes. "
        "Hit the portals on the beat to maintain your speed."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 5000

        # Lanes
        self.LANE_YS = [100, 170, 240, 310]
        self.NUM_LANES = len(self.LANE_YS)

        # Player
        self.PLAYER_X = 120
        self.PLAYER_WIDTH = 15
        self.PLAYER_HEIGHT = 40
        self.PLAYER_INTERP_SPEED = 0.2

        # Tempo & Speed
        self.INITIAL_BPM = 60.0
        self.BPM_INCREASE_RATE = 0.05
        self.BPM_INCREASE_INTERVAL = 200
        self.TEMPO_TO_SPEED_RATIO = 0.1 # Relates BPM to pixels/frame
        self.SPEED_DECAY = 0.995
        self.PORTAL_HIT_SPEED_BOOST = 3.0
        self.PORTAL_MISS_SPEED_PENALTY = 1.0

        # Portals
        self.PORTAL_RADIUS = 30
        self.PORTAL_SPAWN_X_OFFSET = self.WIDTH + 100
        self.PORTAL_HIT_WINDOW_BEATS = 0.2 # +/- this value around the beat
        self.PORTAL_COMPLEXITY_INTERVAL = 500

        # Colors (Vibrant & High Contrast)
        self.COLOR_BG_TOP = (10, 5, 25)
        self.COLOR_BG_BOTTOM = (25, 10, 40)
        self.COLOR_PLAYER = (0, 255, 255) # Bright Cyan
        self.COLOR_PLAYER_GLOW = (100, 255, 255)
        self.PORTAL_COLORS = [
            (255, 0, 128),  # Magenta
            (0, 255, 128),  # Spring Green
            (255, 128, 0),  # Orange
            (128, 0, 255),  # Violet
        ]
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.COLOR_BEAT_BAR = (255, 255, 255)
        self.COLOR_SUCCESS = (255, 255, 100)
        self.COLOR_FAIL = (255, 50, 50)

        # --- Gymnasium Setup ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)

        # --- State Variables ---
        # These are initialized in reset()
        self.steps = None
        self.score = None
        self.camera_x = None
        self.player_lane = None
        self.player_visual_y = None
        self.player_speed = None
        self.tempo_bpm = None
        self.time_since_last_beat = None
        self.measure_beat = None
        self.portals = None
        self.particles = None
        self.last_reward_score_milestone = None
        self.unlocked_songs = None
        self.action_feedback = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0.0
        self.camera_x = 0.0
        self.player_lane = 1
        self.player_visual_y = self.LANE_YS[self.player_lane]
        self.tempo_bpm = self.INITIAL_BPM
        
        initial_speed = self.TEMPO_TO_SPEED_RATIO * self.tempo_bpm
        self.player_speed = initial_speed

        self.time_since_last_beat = 0.0
        self.measure_beat = 0
        self.portals = []
        self.particles = []
        self.last_reward_score_milestone = 0
        self.unlocked_songs = 1
        self.action_feedback = None

        # Pre-spawn some initial portals
        for i in range(5):
            self._spawn_portal(spawn_x_offset=self.WIDTH / 2 + i * 200)

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement = action[0]
        # space_held = action[1] == 1 # Unused
        # shift_held = action[2] == 1 # Unused

        self.steps += 1
        reward = 0
        self.action_feedback = None

        # --- Update Game Logic ---
        self._update_beat_and_tempo()
        
        action_reward = self._handle_player_action(movement)
        reward += action_reward

        self._update_player_state()
        self._update_portals()
        self._update_particles()
        
        # --- Calculate Rewards ---
        # Continuous speed reward
        target_speed = self.TEMPO_TO_SPEED_RATIO * self.tempo_bpm
        speed_ratio = self.player_speed / target_speed if target_speed > 0 else 0
        if speed_ratio >= 0.9:
            reward += 1.0  # At pace
        elif speed_ratio >= 0.7:
            reward += -0.1 # Slightly behind
        else:
            reward += -1.0 # Significantly behind
        
        # Distance milestone reward
        if self.score // 100 > self.last_reward_score_milestone:
            milestone_diff = (self.score // 100) - self.last_reward_score_milestone
            reward += 10.0 * milestone_diff
            self.last_reward_score_milestone = self.score // 100
            # # Sound: Milestone reached!
            self._spawn_particles(self.PLAYER_X, self.player_visual_y, self.COLOR_SUCCESS, 50, life=40, speed=4)

        # --- Check Termination ---
        terminated = False
        if speed_ratio < 0.5 and self.steps > 60: # Grace period at start
            terminated = True
            reward = -50.0 # Terminal penalty
        
        truncated = False
        if self.steps >= self.MAX_STEPS:
            truncated = True
            reward = 50.0 # Terminal reward for finishing

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info(),
        )

    def _update_beat_and_tempo(self):
        # Update tempo
        if self.steps > 0 and self.steps % self.BPM_INCREASE_INTERVAL == 0:
            self.tempo_bpm += self.BPM_INCREASE_RATE

        # Update beat counter
        seconds_per_beat = 60.0 / self.tempo_bpm
        self.time_since_last_beat += 1.0 / self.FPS
        if self.time_since_last_beat >= seconds_per_beat:
            self.time_since_last_beat -= seconds_per_beat
            self.measure_beat = (self.measure_beat + 1) % 4
            # # Sound: Metronome tick
            # Activate portals on their beat
            for p in self.portals:
                p['is_active_now'] = (self.measure_beat == p['active_beat'])

    def _handle_player_action(self, movement):
        reward = 0
        target_lane_idx = movement - 1
        
        if 0 <= target_lane_idx < self.NUM_LANES:
            # Player chose a lane
            self.player_lane = target_lane_idx
            
            # Find closest portal in that lane
            hittable_portal = None
            min_dist = float('inf')
            for p in self.portals:
                if p['lane'] == target_lane_idx:
                    dist = abs(p['x'] - self.PLAYER_X)
                    if dist < min_dist and dist < self.PORTAL_RADIUS * 2:
                        min_dist = dist
                        hittable_portal = p
            
            if hittable_portal:
                beat_progress = self.time_since_last_beat / (60.0 / self.tempo_bpm)
                is_on_beat = abs(beat_progress - 1.0) < self.PORTAL_HIT_WINDOW_BEATS or beat_progress < self.PORTAL_HIT_WINDOW_BEATS

                if hittable_portal['is_active_now'] and is_on_beat:
                    # Successful hit
                    # # Sound: Portal success!
                    self.player_speed = max(self.player_speed, self.TEMPO_TO_SPEED_RATIO * self.tempo_bpm) + self.PORTAL_HIT_SPEED_BOOST
                    hittable_portal['used'] = True
                    reward += 5.0 # Event reward
                    self.action_feedback = {'type': 'success', 'pos': (hittable_portal['x'], self.LANE_YS[hittable_portal['lane']])}
                    self._spawn_particles(self.PLAYER_X, self.player_visual_y, hittable_portal['color'], 30, life=30, speed=5)
                else:
                    # Missed timing
                    # # Sound: Portal miss
                    self.player_speed -= self.PORTAL_MISS_SPEED_PENALTY
                    reward -= 2.0
                    self.action_feedback = {'type': 'fail', 'pos': (self.PLAYER_X, self.player_visual_y)}
            else:
                # Chose an empty lane
                # # Sound: Whoosh (empty)
                self.player_speed -= self.PORTAL_MISS_SPEED_PENALTY / 2
        return reward

    def _update_player_state(self):
        # Natural speed decay
        self.player_speed *= self.SPEED_DECAY
        self.player_speed = max(0, self.player_speed)
        
        # Update world scroll and score
        self.camera_x += self.player_speed
        self.score += self.player_speed * 0.01

        # Interpolate visual Y position for smooth lane changes
        target_y = self.LANE_YS[self.player_lane]
        self.player_visual_y += (target_y - self.player_visual_y) * self.PLAYER_INTERP_SPEED
        
        # Spawn trail particles
        if self.steps % 2 == 0:
            self._spawn_particles(self.PLAYER_X, self.player_visual_y, self.COLOR_PLAYER_GLOW, 1, life=20, speed=0.5, size=3)

    def _update_portals(self):
        # Move and remove old portals
        for p in self.portals[:]:
            p['x'] -= self.player_speed
            if p['x'] < -self.PORTAL_RADIUS:
                self.portals.remove(p)

        # Spawn new portals
        last_portal_x = 0
        if self.portals:
            last_portal_x = max(p['x'] for p in self.portals)
        
        if last_portal_x < self.camera_x + self.WIDTH:
            self._spawn_portal(spawn_x_offset=last_portal_x + random.uniform(250, 400))
    
    def _spawn_portal(self, spawn_x_offset):
        # Increase complexity over time
        if self.steps < self.PORTAL_COMPLEXITY_INTERVAL:
            active_beat_options = [0, 1, 2, 3] # Easy: any beat can be active
        elif self.steps < self.PORTAL_COMPLEXITY_INTERVAL * 2:
            active_beat_options = [0, 2] # Medium: only on/off beats
        else:
            active_beat_options = [0] # Hard: only on the downbeat

        portal = {
            'x': spawn_x_offset,
            'lane': self.np_random.integers(0, self.NUM_LANES),
            'color': random.choice(self.PORTAL_COLORS),
            'active_beat': random.choice(active_beat_options),
            'is_active_now': False,
            'used': False,
            'spawn_time': self.steps
        }
        self.portals.append(portal)

    def _update_particles(self):
        for p in self.particles[:]:
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
                continue
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['vx'] *= 0.95
            p['vy'] *= 0.95

    def _spawn_particles(self, x, y, color, count, life, speed, size=2):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            s = self.np_random.uniform(speed * 0.5, speed)
            self.particles.append({
                'x': x, 'y': y,
                'vx': math.cos(angle) * s - self.player_speed, # Offset by player speed
                'vy': math.sin(angle) * s,
                'life': self.np_random.integers(life // 2, life),
                'color': color,
                'size': size,
            })

    def _get_observation(self):
        self._render_to_screen()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        if self.score // 1000 > self.unlocked_songs - 1:
            self.unlocked_songs = int(self.score // 1000) + 1

        return {
            "score": self.score,
            "steps": self.steps,
            "tempo_bpm": self.tempo_bpm,
            "player_speed": self.player_speed,
            "unlocked_songs": self.unlocked_songs,
        }

    def _render_to_screen(self):
        # --- Draw Background ---
        for y in range(self.HEIGHT):
            interp = y / self.HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

        # --- Draw Portals ---
        beat_progress = self.time_since_last_beat / (60.0 / self.tempo_bpm)
        pulse = (1.0 - math.cos(beat_progress * 2 * math.pi)) / 2.0 # Smooth 0-1-0 pulse

        for p in sorted(self.portals, key=lambda pt: pt['spawn_time']):
            if p['used']: continue
            
            screen_x = int(p['x'] - self.camera_x)
            screen_y = int(self.LANE_YS[p['lane']])
            
            # Glow effect for active portals
            is_active = (self.measure_beat == p['active_beat'])
            radius = self.PORTAL_RADIUS
            if is_active:
                glow_radius = int(radius * (1.2 + pulse * 0.4))
                glow_alpha = 100 + pulse * 100
                pygame.gfxdraw.filled_circle(self.screen, screen_x, screen_y, glow_radius, (*p['color'], glow_alpha))
            
            # Main circle
            main_radius = int(radius * (0.9 + pulse * 0.1))
            pygame.gfxdraw.aacircle(self.screen, screen_x, screen_y, main_radius, p['color'])
            pygame.gfxdraw.filled_circle(self.screen, screen_x, screen_y, main_radius, (*p['color'], 150))
            
            # Inner core
            core_radius = int(main_radius * 0.3)
            pygame.gfxdraw.filled_circle(self.screen, screen_x, screen_y, core_radius, (255, 255, 255))

        # --- Draw Particles ---
        for p in self.particles:
            screen_x = int(p['x'] - self.camera_x)
            screen_y = int(p['y'])
            alpha = max(0, 255 * (p['life'] / 20))
            color = (*p['color'], alpha)
            size = int(p['size'] * (p['life'] / 20))
            if size > 0:
                pygame.draw.rect(self.screen, color, (screen_x, screen_y, size, size))
        
        # --- Draw Action Feedback ---
        if self.action_feedback:
            x, y = self.action_feedback['pos']
            screen_x, screen_y = int(x - self.camera_x), int(y)
            if self.action_feedback['type'] == 'success':
                pygame.gfxdraw.aacircle(self.screen, screen_x, screen_y, 40, self.COLOR_SUCCESS)
            elif self.action_feedback['type'] == 'fail':
                pygame.gfxdraw.aacircle(self.screen, screen_x, screen_y, 20, self.COLOR_FAIL)


        # --- Draw Player ---
        player_rect = pygame.Rect(
            self.PLAYER_X - self.PLAYER_WIDTH // 2,
            self.player_visual_y - self.PLAYER_HEIGHT // 2,
            self.PLAYER_WIDTH,
            self.PLAYER_HEIGHT
        )
        # Glow
        glow_rect = player_rect.inflate(10, 10)
        shape_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(shape_surf, (*self.COLOR_PLAYER_GLOW, 80), (0, 0, *glow_rect.size), border_radius=8)
        self.screen.blit(shape_surf, glow_rect.topleft)
        # Body
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=5)

        # --- Draw UI ---
        # Score
        score_text = self.font_large.render(f"SCORE: {int(self.score):06d}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Tempo
        tempo_text = self.font_small.render(f"BPM: {self.tempo_bpm:.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(tempo_text, (self.WIDTH // 2 - tempo_text.get_width() // 2, 10))
        
        # Beat Bar
        bar_width = 100
        bar_height = 5
        bar_x = self.WIDTH // 2 - bar_width // 2
        bar_y = 35
        pygame.draw.rect(self.screen, (50, 50, 80), (bar_x, bar_y, bar_width, bar_height))
        progress_width = bar_width * beat_progress
        pygame.draw.rect(self.screen, self.COLOR_BEAT_BAR, (bar_x, bar_y, progress_width, bar_height))
        
        # Beat indicator flash
        if pulse > 0.95:
             pygame.draw.rect(self.screen, self.COLOR_BEAT_BAR, (bar_x - 5, bar_y - 2, bar_width + 10, bar_height + 4), 1)

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # Manual play loop
    # Re-enable display for manual testing
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Rhythm Runner")
    clock = pygame.time.Clock()
    
    done = False
    total_reward = 0
    
    print("\n--- Controls ---")
    print("1, 2, 3, 4: Select lane")
    print("Q: Quit")
    print("----------------\n")
    
    while not done:
        action = env.action_space.sample() # Default to random action
        action[0] = 0 # No-op by default

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    done = True
                if event.key == pygame.K_1:
                    action[0] = 1
                if event.key == pygame.K_2:
                    action[0] = 2
                if event.key == pygame.K_3:
                    action[0] = 3
                if event.key == pygame.K_4:
                    action[0] = 4
                if event.key == pygame.K_r: # Manual reset
                    obs, info = env.reset()
                    total_reward = 0
                    print("--- Game Reset ---")


        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            print(f"Episode Finished. Final Score: {info['score']:.2f}, Total Reward: {total_reward:.2f}")
            done = True
            
        # Display the observation from the environment
        # A bit more direct since we have the screen in the env
        env._render_to_screen()
        # The observation is transposed, so we need to convert it back for display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.FPS)
        
    env.close()