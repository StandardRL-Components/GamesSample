import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T18:02:37.156513
# Source Brief: brief_02539.md
# Brief Index: 2539
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Jump between colored platforms to reach the exit. Match platform colors to gain energy, but watch out for patrolling guards. Use your energy to craft a temporary shield for protection."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to jump between platforms. Press space to activate a shield."
    )
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 2000

    # Colors
    COLOR_BG = (26, 26, 46) # Dark blue/purple
    COLOR_WHITE = (240, 240, 240)
    COLOR_PLATFORM_FILLS = [(50, 20, 20), (20, 50, 20), (20, 20, 50)]
    COLOR_PLATFORM_OUTLINES = [(255, 80, 80), (80, 255, 80), (80, 80, 255)] # Red, Green, Blue
    COLOR_EXIT = (255, 255, 0)
    COLOR_GUARD = (255, 255, 255)
    COLOR_GUARD_DETECT = (255, 0, 0, 100)
    COLOR_SHIELD = (0, 255, 255)
    COLOR_PATH = (60, 60, 80)
    COLOR_ENERGY_BAR = (40, 200, 60)
    COLOR_ENERGY_BAR_BG = (80, 80, 80)

    # Player
    PLAYER_JUMP_SPEED = 0.05 # Progress per frame
    PLAYER_JUMP_ARC_HEIGHT = 80
    PLAYER_START_ENERGY = 100.0

    # Energy & Shield
    ENERGY_JUMP_COST = 1.0
    ENERGY_GAIN_MATCH = 25.0
    ENERGY_LOSS_MISMATCH = 15.0
    SHIELD_COST = 30.0
    SHIELD_DURATION = 150 # steps

    # Guards
    GUARD_SIZE = 16
    GUARD_DETECT_RADIUS = 80
    INITIAL_GUARD_SPEED = 0.75

    # Platforms
    PLATFORM_SIZE = 24
    NUM_PLATFORMS = 12

    # Rewards
    REWARD_STEP = -0.01
    REWARD_MATCH = 1.0
    REWARD_MISMATCH = -0.5
    REWARD_CRAFT_SHIELD = 0.5
    REWARD_NEAR_GUARD = -1.0
    REWARD_WIN = 100.0
    REWARD_LOSE = -100.0

    # Class attribute for persistent state
    persistent_guard_speed = INITIAL_GUARD_SPEED

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 20, bold=True)
        self.render_mode = render_mode

        # Initialize state variables to avoid attribute errors
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.platforms = []
        self.start_platform_idx = 0
        self.exit_platform_idx = 0
        self.guards = []
        self.player_pos = pygame.math.Vector2(0, 0)
        self.player_platform_idx = 0
        self.player_energy = 0.0
        self.player_shield_active = False
        self.player_shield_timer = 0
        self.player_jump_state = {'is_jumping': False}
        self.last_space_held = False
        self.particles = []
        self.current_reward = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.current_reward = 0.0
        self.last_space_held = False
        self.particles = []

        self._generate_level()

        self.player_platform_idx = self.start_platform_idx
        self.player_pos = pygame.math.Vector2(self.platforms[self.player_platform_idx]['pos'])
        self.player_energy = self.PLAYER_START_ENERGY
        self.player_shield_active = False
        self.player_shield_timer = 0
        self.player_jump_state = {
            'is_jumping': False,
            'start_pos': None,
            'end_pos': None,
            'progress': 0.0,
            'target_idx': -1
        }
        
        # Difficulty scaling
        if self.steps > 0 and self.steps % 200 == 0:
            GameEnv.persistent_guard_speed = min(2.0, GameEnv.persistent_guard_speed + 0.05)

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.current_reward = self.REWARD_STEP
        self.steps += 1

        self._handle_actions(action)
        self._update_game_state()
        self._check_events_and_termination()
        
        self.score += self.current_reward
        terminated = self.game_over
        truncated = self.steps >= self.MAX_STEPS

        return (
            self._get_observation(),
            self.current_reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_actions(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1

        # Handle shield activation (on button press)
        if space_held and not self.last_space_held:
            if self.player_energy >= self.SHIELD_COST and not self.player_shield_active:
                self.player_energy -= self.SHIELD_COST
                self.player_shield_active = True
                self.player_shield_timer = self.SHIELD_DURATION
                self.current_reward += self.REWARD_CRAFT_SHIELD
                self._create_shield_particles()
                # sfx: shield_activate.wav
        self.last_space_held = space_held

        # Handle jump action
        if movement != 0 and not self.player_jump_state['is_jumping']:
            target_idx = self._find_target_platform(movement)
            if target_idx is not None:
                self.player_jump_state = {
                    'is_jumping': True,
                    'start_pos': self.player_pos.copy(),
                    'end_pos': pygame.math.Vector2(self.platforms[target_idx]['pos']),
                    'progress': 0.0,
                    'target_idx': target_idx
                }
                self.player_platform_idx = None
                self.player_energy -= self.ENERGY_JUMP_COST
                # sfx: jump.wav

    def _update_game_state(self):
        self._update_player_jump()
        self._update_guards()
        self._update_shield()
        self._update_particles()

    def _check_events_and_termination(self):
        # Check guard proximity
        is_near_guard = False
        for guard in self.guards:
            if self.player_pos.distance_to(guard['pos']) < self.GUARD_DETECT_RADIUS:
                if not self.player_shield_active:
                    self.current_reward += self.REWARD_NEAR_GUARD
                    is_near_guard = True
                    if self.player_pos.distance_to(guard['pos']) < self.GUARD_SIZE:
                        self.game_over = True
                        self.current_reward += self.REWARD_LOSE
                        # sfx: lose_collision.wav
                        return
                break
        
        # Check for max steps is handled in step() return
        if self.steps >= self.MAX_STEPS:
            self.game_over = True

        # Check for energy depletion
        if self.player_energy <= 0:
            self.player_energy = 0
            self.game_over = True
            self.current_reward += self.REWARD_LOSE
            # sfx: lose_energy.wav

    def _generate_level(self):
        max_retries = 100
        for _ in range(max_retries):
            self.platforms = []
            
            # Place start and exit platforms
            start_pos = (self.np_random.integers(40, 101), self.np_random.integers(50, self.HEIGHT - 49))
            exit_pos = (self.np_random.integers(self.WIDTH - 100, self.WIDTH - 39), self.np_random.integers(50, self.HEIGHT - 49))
            
            self.platforms.append(self._create_platform(start_pos, 0))
            self.start_platform_idx = 0
            self.platforms.append(self._create_platform(exit_pos, 0)) # Color doesn't matter for exit
            self.exit_platform_idx = 1
            
            # Place other platforms
            for _ in range(self.NUM_PLATFORMS - 2):
                while True:
                    pos = (self.np_random.integers(30, self.WIDTH - 29), self.np_random.integers(30, self.HEIGHT - 29))
                    if not any(pygame.Rect(p['pos'][0]-self.PLATFORM_SIZE, p['pos'][1]-self.PLATFORM_SIZE, self.PLATFORM_SIZE*2, self.PLATFORM_SIZE*2).collidepoint(pos) for p in self.platforms):
                        self.platforms.append(self._create_platform(pos, self.np_random.integers(0, 3)))
                        break
            
            if self._is_level_solvable():
                break
        else: # If loop finishes without break
            # Fallback to a guaranteed solvable but simple layout if generation fails
            print("Warning: Level generation failed, using fallback.")
            self._generate_fallback_level()

        self._generate_guards()

    def _is_level_solvable(self):
        adj = {i: [] for i in range(len(self.platforms))}
        for i, p1 in enumerate(self.platforms):
            for j, p2 in enumerate(self.platforms):
                if i == j: continue
                # Simple connectivity: if within a certain distance
                dist = pygame.math.Vector2(p1['pos']).distance_to(p2['pos'])
                if dist < self.WIDTH / 3:
                     adj[i].append(j)

        q = deque([self.start_platform_idx])
        visited = {self.start_platform_idx}
        while q:
            u = q.popleft()
            if u == self.exit_platform_idx:
                return True
            for v in adj.get(u, []):
                if v not in visited:
                    visited.add(v)
                    q.append(v)
        return False
    
    def _generate_fallback_level(self):
        self.platforms = []
        self.platforms.append(self._create_platform((80, self.HEIGHT//2), 0))
        self.start_platform_idx = 0
        self.platforms.append(self._create_platform((self.WIDTH-80, self.HEIGHT//2), 0))
        self.exit_platform_idx = 1
        self.platforms.append(self._create_platform((self.WIDTH//2, self.HEIGHT//2), 1))

    def _create_platform(self, pos, color_idx):
        return {
            'pos': pos,
            'color_idx': color_idx,
            'rect': pygame.Rect(pos[0] - self.PLATFORM_SIZE // 2, pos[1] - self.PLATFORM_SIZE // 2, self.PLATFORM_SIZE, self.PLATFORM_SIZE)
        }

    def _generate_guards(self):
        self.guards = []
        num_guards = 1 + self.steps // 500 # More guards over time
        for _ in range(num_guards):
            path_type = self.np_random.choice(['horizontal', 'vertical', 'box'])
            if path_type == 'horizontal':
                y = self.np_random.integers(self.GUARD_SIZE, self.HEIGHT - self.GUARD_SIZE + 1)
                x1 = self.np_random.integers(self.GUARD_SIZE, self.WIDTH // 2 - 49)
                x2 = self.np_random.integers(self.WIDTH // 2 + 50, self.WIDTH - self.GUARD_SIZE + 1)
                path = [(x1, y), (x2, y)]
            elif path_type == 'vertical':
                x = self.np_random.integers(self.GUARD_SIZE, self.WIDTH - self.GUARD_SIZE + 1)
                y1 = self.np_random.integers(self.GUARD_SIZE, self.HEIGHT // 2 - 49)
                y2 = self.np_random.integers(self.HEIGHT // 2 + 50, self.HEIGHT - self.GUARD_SIZE + 1)
                path = [(x, y1), (x, y2)]
            else: # box
                cx = self.np_random.integers(100, self.WIDTH - 99)
                cy = self.np_random.integers(100, self.HEIGHT - 99)
                w, h = self.np_random.integers(80, 151), self.np_random.integers(80, 151)
                path = [(cx-w, cy-h), (cx+w, cy-h), (cx+w, cy+h), (cx-w, cy+h)]

            self.guards.append({
                'path': [pygame.math.Vector2(p) for p in path],
                'path_idx': 0,
                'pos': pygame.math.Vector2(path[0]),
                'speed': GameEnv.persistent_guard_speed
            })

    def _find_target_platform(self, movement_action):
        current_pos = self.platforms[self.player_platform_idx]['pos']
        best_target = None
        min_dist = float('inf')

        for i, p in enumerate(self.platforms):
            if i == self.player_platform_idx:
                continue

            vec = pygame.math.Vector2(p['pos']) - pygame.math.Vector2(current_pos)
            if vec.length() == 0: continue

            angle = math.degrees(math.atan2(-vec.y, vec.x)) # Pygame y is inverted
            
            # Check if platform is in the correct quadrant
            correct_dir = False
            if movement_action == 1 and 45 <= angle < 135: correct_dir = True  # Up
            elif movement_action == 2 and -135 <= angle < -45: correct_dir = True # Down
            elif movement_action == 3 and (135 <= angle <= 180 or -180 <= angle < -135): correct_dir = True # Left
            elif movement_action == 4 and -45 <= angle < 45: correct_dir = True # Right

            if correct_dir:
                dist = vec.length()
                if dist < min_dist:
                    min_dist = dist
                    best_target = i
        
        return best_target

    def _update_player_jump(self):
        if not self.player_jump_state['is_jumping']:
            return

        state = self.player_jump_state
        state['progress'] += self.PLAYER_JUMP_SPEED
        
        # Add trail particle
        if self.steps % 2 == 0:
            self._create_trail_particle()

        if state['progress'] >= 1.0:
            # Landed
            state['is_jumping'] = False
            state['progress'] = 1.0
            self.player_platform_idx = state['target_idx']
            self.player_pos = state['end_pos'].copy()

            # Check for win condition
            if self.player_platform_idx == self.exit_platform_idx:
                self.game_over = True
                self.current_reward += self.REWARD_WIN
                # sfx: win.wav
                GameEnv.persistent_guard_speed = min(2.0, GameEnv.persistent_guard_speed + 0.05)

            else: # Normal landing
                start_color = self.platforms[self.start_platform_idx]['color_idx']
                target_color = self.platforms[self.player_platform_idx]['color_idx']
                
                if start_color == target_color:
                    self.player_energy = min(self.PLAYER_START_ENERGY, self.player_energy + self.ENERGY_GAIN_MATCH)
                    self.current_reward += self.REWARD_MATCH
                    # sfx: match.wav
                else:
                    self.player_energy -= self.ENERGY_LOSS_MISMATCH
                    self.current_reward += self.REWARD_MISMATCH
                    # sfx: mismatch.wav
            
            self._create_landing_particles()
        else:
            # Interpolate position
            progress = state['progress']
            self.player_pos = state['start_pos'].lerp(state['end_pos'], progress)
            # Add arc
            arc = math.sin(progress * math.pi) * self.PLAYER_JUMP_ARC_HEIGHT
            self.player_pos.y -= arc

    def _update_guards(self):
        for guard in self.guards:
            target_node_idx = (guard['path_idx'] + 1) % len(guard['path'])
            target_pos = guard['path'][target_node_idx]
            
            direction = (target_pos - guard['pos'])
            if direction.length() < guard['speed']:
                guard['pos'] = target_pos
                guard['path_idx'] = target_node_idx
            else:
                guard['pos'] += direction.normalize() * guard['speed']

    def _update_shield(self):
        if self.player_shield_active:
            self.player_shield_timer -= 1
            if self.player_shield_timer <= 0:
                self.player_shield_active = False
                # sfx: shield_deactivate.wav

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['life'] -= 1
            p['pos'] += p['vel']
            p['size'] = max(0, p['size'] - p['decay'])
    
    def _create_particle(self, pos, vel, life, color, size, decay=0.1):
        self.particles.append({'pos': pygame.math.Vector2(pos), 'vel': pygame.math.Vector2(vel), 'life': life, 'color': color, 'size': size, 'decay': decay})

    def _create_landing_particles(self):
        if self.player_platform_idx is None: return
        pos = self.platforms[self.player_platform_idx]['pos']
        color = self.COLOR_PLATFORM_OUTLINES[self.platforms[self.player_platform_idx]['color_idx']]
        for _ in range(20):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 3)
            vel = (math.cos(angle) * speed, math.sin(angle) * speed)
            self._create_particle(pos, vel, self.np_random.integers(15, 31), color, random.uniform(2, 5))

    def _create_shield_particles(self):
        for _ in range(40):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(2, 4)
            vel = (math.cos(angle) * speed, math.sin(angle) * speed)
            self._create_particle(self.player_pos, vel, self.np_random.integers(20, 41), self.COLOR_SHIELD, random.uniform(1, 4))
    
    def _create_trail_particle(self):
        if not self.player_jump_state['is_jumping']: return
        start_idx = self.start_platform_idx if self.player_platform_idx is None else self.player_platform_idx
        color = self.COLOR_PLATFORM_OUTLINES[self.platforms[start_idx]['color_idx']]
        vel = (random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5))
        self._create_particle(self.player_pos, vel, 20, color, 4, 0.2)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "energy": self.player_energy}

    def _render_game(self):
        self._render_paths()
        self._render_platforms()
        self._render_jump_arc()
        self._render_guards()
        self._render_particles()
        self._render_player()

    def _render_paths(self):
        for guard in self.guards:
            if len(guard['path']) > 1:
                pygame.draw.lines(self.screen, self.COLOR_PATH, True, [p for p in guard['path']], 1)

    def _render_platforms(self):
        for i, p in enumerate(self.platforms):
            is_exit = (i == self.exit_platform_idx)
            color_fill = self.COLOR_BG if is_exit else self.COLOR_PLATFORM_FILLS[p['color_idx']]
            color_outline = self.COLOR_EXIT if is_exit else self.COLOR_PLATFORM_OUTLINES[p['color_idx']]
            
            pygame.draw.rect(self.screen, color_fill, p['rect'])
            pygame.draw.rect(self.screen, color_outline, p['rect'], 2)

            if is_exit: # Glow effect for exit
                for i in range(5, 0, -1):
                    alpha = 150 - i * 25
                    color = (*self.COLOR_EXIT, alpha)
                    radius = self.PLATFORM_SIZE // 2 + i * 2
                    pygame.gfxdraw.aacircle(self.screen, int(p['pos'][0]), int(p['pos'][1]), radius, color)

    def _render_jump_arc(self):
        if not self.player_jump_state['is_jumping']:
            start_idx = self.player_platform_idx
            if start_idx is not None:
                start_color = self.COLOR_PLATFORM_OUTLINES[self.platforms[start_idx]['color_idx']]
                for move_action in range(1, 5):
                    target_idx = self._find_target_platform(move_action)
                    if target_idx is not None:
                        start_pos = self.platforms[start_idx]['pos']
                        end_pos = self.platforms[target_idx]['pos']
                        target_color = self.COLOR_PLATFORM_OUTLINES[self.platforms[target_idx]['color_idx']]
                        color = start_color if start_color == target_color else self.COLOR_WHITE
                        pygame.draw.aaline(self.screen, (*color, 60), start_pos, end_pos)

    def _render_guards(self):
        for guard in self.guards:
            pos = (int(guard['pos'].x), int(guard['pos'].y))
            size = self.GUARD_SIZE
            points = [
                (pos[0], pos[1] - size // 2),
                (pos[0] - size // 2, pos[1] + size // 2),
                (pos[0] + size // 2, pos[1] + size // 2),
            ]
            pygame.draw.polygon(self.screen, self.COLOR_GUARD, points)
            pygame.draw.aalines(self.screen, self.COLOR_GUARD, True, points)

            # Detection radius
            if self.player_pos.distance_to(guard['pos']) < self.GUARD_DETECT_RADIUS and not self.player_shield_active:
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.GUARD_DETECT_RADIUS, self.COLOR_GUARD_DETECT)
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.GUARD_DETECT_RADIUS, self.COLOR_GUARD_DETECT)


    def _render_player(self):
        pos = (int(self.player_pos.x), int(self.player_pos.y))
        
        # Player core
        start_idx = self.start_platform_idx if self.player_jump_state['is_jumping'] else self.player_platform_idx
        player_color = self.COLOR_PLATFORM_OUTLINES[self.platforms[start_idx]['color_idx']]
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 8, player_color)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 8, player_color)
        
        # Shield effect
        if self.player_shield_active:
            pulse = (math.sin(self.steps * 0.2) + 1) / 2 # 0 to 1
            radius = int(18 + pulse * 4)
            alpha = int(80 + (self.player_shield_timer / self.SHIELD_DURATION) * 70)
            color = (*self.COLOR_SHIELD, alpha)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, (*self.COLOR_SHIELD, alpha + 50))


    def _render_particles(self):
        for p in self.particles:
            if p['size'] > 1:
                pygame.draw.circle(self.screen, p['color'], (int(p['pos'].x), int(p['pos'].y)), int(p['size']))

    def _render_ui(self):
        # Score
        score_text = self.font.render(f"SCORE: {int(self.score)}", True, self.COLOR_WHITE)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))

        # Energy Bar
        bar_width = 200
        bar_height = 15
        energy_ratio = max(0, self.player_energy / self.PLAYER_START_ENERGY)
        
        bg_rect = pygame.Rect(10, 10, bar_width, bar_height)
        pygame.draw.rect(self.screen, self.COLOR_ENERGY_BAR_BG, bg_rect)
        
        fill_rect = pygame.Rect(10, 10, int(bar_width * energy_ratio), bar_height)
        pygame.draw.rect(self.screen, self.COLOR_ENERGY_BAR, fill_rect)
        
        pygame.draw.rect(self.screen, self.COLOR_WHITE, bg_rect, 1)

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # The validation logic has been removed as it was for internal testing
    # and is not needed for the final runnable script.
    # The main execution block is for manual play and demonstration.
    
    # Set the video driver for local rendering
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    done = False
    
    pygame.display.set_caption("High-Voltage Escape")
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    clock = pygame.time.Clock()
    
    total_reward = 0
    
    action = env.action_space.sample()
    action[0] = 0 # No movement initially
    action[1] = 0 # No space
    action[2] = 0 # No shift

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        keys = pygame.key.get_pressed()
        action[0] = 0 # No-op
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        action[1] = 1 if keys[pygame.K_SPACE] else 0
        action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Blit the observation from the environment to the display screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(GameEnv.FPS)

        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Info: {info}")
            total_reward = 0
            obs, info = env.reset()
            # Pause for a moment before starting next episode
            pygame.time.wait(1000)
            if done: break # Exit loop if quit event was received during pause

    env.close()