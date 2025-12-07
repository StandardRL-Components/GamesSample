import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:36:32.263904
# Source Brief: brief_00576.md
# Brief Index: 576
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
        "Infiltrate a high-tech facility as a cyber ninja. Evade enemy patrols, flip gravity to navigate, and eliminate all targets to complete your mission."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move. Press space to attack nearby targets. Press shift to flip gravity."
    )
    auto_advance = True

    # --- CONSTANTS ---
    # Colors
    COLOR_BG = (15, 15, 25)
    COLOR_GRID = (30, 30, 50)
    COLOR_PLAYER = (0, 255, 150)
    COLOR_PLAYER_GLOW = (0, 255, 150, 50)
    COLOR_TARGET = (255, 50, 50)
    COLOR_TARGET_GLOW = (255, 50, 50, 50)
    COLOR_VISION_CONE = (200, 50, 50, 30)
    COLOR_VISION_CONE_DETECT = (255, 100, 100, 60)
    COLOR_PATH = (80, 80, 100, 100)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_GRAVITY_COOLDOWN_BG = (40, 40, 60)
    COLOR_GRAVITY_COOLDOWN_FG = (50, 150, 255)
    COLOR_DETECTION_BG = (60, 40, 40)
    COLOR_DETECTION_FG = (255, 80, 80)
    
    # Screen Dimensions
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Game Parameters
    FPS = 30
    MAX_STEPS = 1000
    GRID_SIZE = 40
    
    # Player
    PLAYER_SIZE = 8
    PLAYER_SPEED = 4.0
    PLAYER_FRICTION = 0.85
    
    # Target
    TARGET_SIZE = 10
    TARGET_VISION_RANGE = 120
    TARGET_ATTACK_RANGE = 25
    
    # Mechanics
    GRAVITY_COOLDOWN_STEPS = 150 # 5 seconds at 30 FPS
    ATTACK_COOLDOWN_STEPS = 15
    DETECTION_LIMIT = 60 # 2 seconds in cone to be detected

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 16, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        
        # State variables are initialized in reset()
        self.level = 1
        self.last_win = False # To track level progression
        
        self.player_pos = None
        self.player_vel = None
        self.gravity_up_is_up = None
        self.gravity_cooldown = None
        self.attack_cooldown = None
        self.detection_level = None
        self.targets = None
        self.particles = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.last_dist_to_target = float('inf')

        # self.reset() is called by the wrapper
        # self.validate_implementation() is for debugging, not needed in final code

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if self.last_win:
            self.level += 1
        self.last_win = False
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player_pos = pygame.Vector2(self.SCREEN_WIDTH / 4, self.SCREEN_HEIGHT / 2)
        self.player_vel = pygame.Vector2(0, 0)
        self.gravity_up_is_up = True
        self.gravity_cooldown = 0
        self.attack_cooldown = 0
        self.detection_level = 0
        self.particles = []
        
        self._generate_level()
        self.last_dist_to_target = self._get_dist_to_nearest_target()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.steps += 1
        step_reward = 0
        
        self._update_cooldowns()
        self._handle_input(action)
        
        self._update_player()
        self._update_targets()
        self._update_particles()
        
        step_reward += self._handle_attacks()
        is_detected, detection_reward = self._check_detection()
        step_reward += detection_reward

        # Proximity reward
        dist_to_target = self._get_dist_to_nearest_target()
        if dist_to_target < self.last_dist_to_target and self.detection_level == 0:
            step_reward += 0.1
        self.last_dist_to_target = dist_to_target
        
        self.score += step_reward
        terminated = self._check_termination(is_detected)
        truncated = self.steps >= self.MAX_STEPS

        if terminated or truncated:
            if len(self.targets) == 0: # Win condition
                step_reward += 50
                self.last_win = True
            elif is_detected: # Loss condition
                step_reward -= 50
            self.score += step_reward
            self.game_over = True
            
        return (
            self._get_observation(),
            step_reward,
            terminated,
            truncated,
            self._get_info()
        )

    # --- UPDATE LOGIC ---
    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Movement
        accel = pygame.Vector2(0, 0)
        if movement == 1: accel.y = -1 # Up
        elif movement == 2: accel.y = 1  # Down
        elif movement == 3: accel.x = -1 # Left
        elif movement == 4: accel.x = 1  # Right
        
        if not self.gravity_up_is_up:
            accel.y *= -1 # Invert vertical movement
            
        if accel.length() > 0:
            self.player_vel += accel.normalize() * self.PLAYER_SPEED

        # Attack
        if space_held and self.attack_cooldown == 0:
            self.attack_cooldown = self.ATTACK_COOLDOWN_STEPS
            self._spawn_particles(self.player_pos, self.COLOR_PLAYER, 20, 2.0, self.TARGET_ATTACK_RANGE)
            # sfx: ninja_attack_swoosh.wav

        # Gravity Flip
        if shift_held and self.gravity_cooldown == 0:
            self.gravity_cooldown = self.GRAVITY_COOLDOWN_STEPS
            self.gravity_up_is_up = not self.gravity_up_is_up
            self._spawn_particles(pygame.Vector2(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2), self.COLOR_GRAVITY_COOLDOWN_FG, 100, 5.0, self.SCREEN_WIDTH, 'wave')
            # sfx: gravity_shift_whoom.wav

    def _update_player(self):
        self.player_pos += self.player_vel
        self.player_vel *= self.PLAYER_FRICTION
        
        # Boundary checks
        self.player_pos.x = np.clip(self.player_pos.x, self.PLAYER_SIZE, self.SCREEN_WIDTH - self.PLAYER_SIZE)
        self.player_pos.y = np.clip(self.player_pos.y, self.PLAYER_SIZE, self.SCREEN_HEIGHT - self.PLAYER_SIZE)

    def _update_targets(self):
        speed_mod = 1.0 + (self.level - 1) * 0.05
        for target in self.targets:
            path_point = target['path'][target['path_index']]
            direction_vec = path_point - target['pos']
            if direction_vec.length() > 0:
                direction = direction_vec.normalize()
                target['pos'] += direction * speed_mod
                target['facing_dir'] = direction
            
            if target['pos'].distance_to(path_point) < 5:
                target['path_index'] = (target['path_index'] + 1) % len(target['path'])

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1

    def _update_cooldowns(self):
        self.gravity_cooldown = max(0, self.gravity_cooldown - 1)
        self.attack_cooldown = max(0, self.attack_cooldown - 1)

    # --- INTERACTION & STATE CHECKS ---
    def _handle_attacks(self):
        reward = 0
        if self.attack_cooldown == self.ATTACK_COOLDOWN_STEPS -1: # Attack just triggered
            for target in self.targets[:]:
                if self.player_pos.distance_to(target['pos']) < self.TARGET_ATTACK_RANGE:
                    self._spawn_particles(target['pos'], self.COLOR_TARGET, 50, 3.0, 40)
                    self.targets.remove(target)
                    reward += 5
                    # sfx: target_eliminated.wav
        return reward

    def _check_detection(self):
        in_cone = False
        reward = 0
        vision_angle_mod = 45 + (self.level - 1) * 1 # degrees
        
        for target in self.targets:
            target['in_sight'] = False
            vec_to_player = self.player_pos - target['pos']
            dist_to_player = vec_to_player.length()
            
            if dist_to_player > 0 and dist_to_player < self.TARGET_VISION_RANGE:
                angle_to_player = math.degrees(target['facing_dir'].angle_to(vec_to_player))
                if abs(angle_to_player) < vision_angle_mod / 2:
                    in_cone = True
                    target['in_sight'] = True
                    break
        
        if in_cone:
            self.detection_level = min(self.DETECTION_LIMIT, self.detection_level + 1)
            reward = -0.5
        else:
            self.detection_level = max(0, self.detection_level - 2)
        
        is_fully_detected = self.detection_level >= self.DETECTION_LIMIT
        return is_fully_detected, reward

    def _check_termination(self, is_detected):
        if len(self.targets) == 0:
            return True # Victory
        if is_detected:
            return True # Failure
        return False

    def _get_dist_to_nearest_target(self):
        if not self.targets:
            return 0
        return min(self.player_pos.distance_to(t['pos']) for t in self.targets)

    # --- LEVEL GENERATION ---
    def _generate_level(self):
        self.targets = []
        num_targets = min(5, 1 + self.level // 2)
        
        for i in range(num_targets):
            path_type = self.np_random.choice(['rect', 'line'])
            if path_type == 'rect':
                w, h = self.np_random.uniform(100, 200), self.np_random.uniform(100, 200)
                x, y = self.np_random.uniform(50, self.SCREEN_WIDTH - w - 50), self.np_random.uniform(50, self.SCREEN_HEIGHT - h - 50)
                path = [pygame.Vector2(x, y), pygame.Vector2(x + w, y), pygame.Vector2(x + w, y + h), pygame.Vector2(x, y + h)]
            else: # line
                x1, y1 = self.np_random.uniform(50, self.SCREEN_WIDTH - 50), self.np_random.uniform(50, self.SCREEN_HEIGHT - 50)
                x2, y2 = self.np_random.uniform(50, self.SCREEN_WIDTH - 50), self.np_random.uniform(50, self.SCREEN_HEIGHT - 50)
                path = [pygame.Vector2(x1, y1), pygame.Vector2(x2, y2)]
            
            start_idx = self.np_random.integers(len(path))
            self.targets.append({
                'pos': pygame.Vector2(path[start_idx]),
                'path': path,
                'path_index': (start_idx + 1) % len(path),
                'facing_dir': pygame.Vector2(1, 0),
                'in_sight': False
            })

    # --- RENDERING ---
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_patrol_paths()
        self._render_targets()
        self._render_player()
        self._render_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for x in range(0, self.SCREEN_WIDTH, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))
        if not self.gravity_up_is_up: # Visual indicator for flipped gravity
            s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            s.fill((50, 50, 150, 15))
            self.screen.blit(s, (0,0))

    def _render_patrol_paths(self):
        for target in self.targets:
            if len(target['path']) > 1:
                pygame.draw.lines(self.screen, self.COLOR_PATH, True if len(target['path']) > 2 else False, [p for p in target['path']], 1)

    def _render_targets(self):
        vision_angle_mod = 45 + (self.level - 1) * 1
        for target in self.targets:
            # Vision cone
            cone_color = self.COLOR_VISION_CONE_DETECT if target['in_sight'] else self.COLOR_VISION_CONE
            if target['facing_dir'].length() > 0:
                p1 = target['pos']
                angle = math.atan2(target['facing_dir'].y, target['facing_dir'].x)
                angle1 = angle - math.radians(vision_angle_mod / 2)
                angle2 = angle + math.radians(vision_angle_mod / 2)
                p2 = target['pos'] + pygame.Vector2(math.cos(angle1), math.sin(angle1)) * self.TARGET_VISION_RANGE
                p3 = target['pos'] + pygame.Vector2(math.cos(angle2), math.sin(angle2)) * self.TARGET_VISION_RANGE
                points = [(int(p1.x), int(p1.y)), (int(p2.x), int(p2.y)), (int(p3.x), int(p3.y))]
                pygame.gfxdraw.aapolygon(self.screen, points, cone_color)
                pygame.gfxdraw.filled_polygon(self.screen, points, cone_color)

            # Target body
            pos = (int(target['pos'].x), int(target['pos'].y))
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.TARGET_SIZE, self.COLOR_TARGET)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.TARGET_SIZE, self.COLOR_TARGET)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.TARGET_SIZE + 5, self.COLOR_TARGET_GLOW)

    def _render_player(self):
        pos = (int(self.player_pos.x), int(self.player_pos.y))
        # Glow effect
        for i in range(4):
             pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.PLAYER_SIZE + i * 3, self.COLOR_PLAYER_GLOW)
        # Player body
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.PLAYER_SIZE, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.PLAYER_SIZE, self.COLOR_PLAYER)

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = p['color'] + (alpha,)
            pos = (int(p['pos'].x), int(p['pos'].y))
            size = int(p['size'] * (p['life'] / p['max_life']))
            if size > 0:
                s = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(s, color, (size, size), size)
                self.screen.blit(s, (pos[0]-size, pos[1]-size))

    def _render_ui(self):
        # Targets remaining
        targets_text = self.font_large.render(f"TARGETS: {len(self.targets)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(targets_text, (10, 10))
        
        # Level
        level_text = self.font_small.render(f"LEVEL: {self.level}", True, self.COLOR_UI_TEXT)
        self.screen.blit(level_text, (10, 40))

        # Gravity Cooldown
        bar_w, bar_h = 100, 15
        grav_x, grav_y = self.SCREEN_WIDTH - bar_w - 10, 10
        grav_ratio = self.gravity_cooldown / self.GRAVITY_COOLDOWN_STEPS
        pygame.draw.rect(self.screen, self.COLOR_GRAVITY_COOLDOWN_BG, (grav_x, grav_y, bar_w, bar_h))
        pygame.draw.rect(self.screen, self.COLOR_GRAVITY_COOLDOWN_FG, (grav_x, grav_y, bar_w * (1-grav_ratio), bar_h))
        grav_text = self.font_small.render("GRAVITY FLIP", True, self.COLOR_UI_TEXT)
        self.screen.blit(grav_text, (grav_x - grav_text.get_width() - 10, grav_y))

        # Detection Meter
        detect_x, detect_y = self.SCREEN_WIDTH/2 - 100, self.SCREEN_HEIGHT - 30
        detect_ratio = self.detection_level / self.DETECTION_LIMIT
        pygame.draw.rect(self.screen, self.COLOR_DETECTION_BG, (detect_x, detect_y, 200, 20))
        pygame.draw.rect(self.screen, self.COLOR_DETECTION_FG, (detect_x, detect_y, 200 * detect_ratio, 20))
        if detect_ratio > 0:
            detect_text = self.font_large.render("! DETECTED !", True, self.COLOR_DETECTION_FG)
            if self.steps % 15 < 7: # Blinking text
                self.screen.blit(detect_text, (detect_x + 200/2 - detect_text.get_width()/2, detect_y - 30))
    
    # --- UTILITY & INFO ---
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
            "targets_remaining": len(self.targets),
            "gravity_cooldown": self.gravity_cooldown,
            "detection_level": self.detection_level,
        }

    def _spawn_particles(self, pos, color, count, max_speed, max_dist, p_type='burst'):
        for _ in range(count):
            if p_type == 'burst':
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(0.5, max_speed)
                vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            elif p_type == 'wave':
                 angle = self.np_random.uniform(0, 2 * math.pi)
                 vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * max_speed
            
            life = self.np_random.integers(15, 30)
            self.particles.append({
                'pos': pygame.Vector2(pos),
                'vel': vel,
                'color': color[:3], # Ensure color is RGB, not RGBA
                'life': life,
                'max_life': life,
                'size': self.np_random.integers(2, 5)
            })

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It will not run in a headless environment
    try:
        os.environ["SDL_VIDEODRIVER"] = "x11"
        env = GameEnv(render_mode="rgb_array")
        obs, info = env.reset()
        
        screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        pygame.display.set_caption("Cyber Ninja Assassin")
        clock = pygame.time.Clock()
        
        running = True
        total_reward = 0.0
        
        while running:
            movement = 0 # No-op
            space = 0
            shift = 0
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            if keys[pygame.K_SPACE]: space = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
                
            action = [movement, space, shift]
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            # Display the observation from the environment
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            if terminated or truncated:
                print(f"Episode finished. Total Reward: {total_reward:.2f}, Info: {info}")
                total_reward = 0.0
                obs, info = env.reset()
                pygame.time.wait(2000) # Pause before starting new game

            clock.tick(GameEnv.FPS)
            
        env.close()
    except pygame.error as e:
        print(f"Could not run interactive test: {e}")
        print("This is expected in a headless environment.")