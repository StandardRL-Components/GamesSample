import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:47:55.116627
# Source Brief: brief_00023.md
# Brief Index: 23
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
        "Stealth platformer where you play as a magnetic ninja. Stick to surfaces, "
        "throw shurikens to stun guards, and use illusions to sneak past them to reach the goal."
    )
    user_guide = (
        "Controls: ←→ to move, ↑ to jump or stick to ceilings, ↓ to stick to the floor. "
        "Press space to throw a shuriken and shift to create a distracting illusion."
    )
    auto_advance = True

    # --- CONSTANTS ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    WORLD_WIDTH = 2560
    FPS = 30
    MAX_STEPS = 5000

    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_GRID = (30, 35, 50)
    COLOR_PLATFORM = (80, 90, 110)
    COLOR_PLATFORM_TOP = (100, 110, 130)
    COLOR_PLAYER = (50, 150, 255)
    COLOR_PLAYER_GLOW = (150, 200, 255)
    COLOR_ENEMY = (255, 80, 80)
    COLOR_ENEMY_VISION = (255, 100, 100)
    COLOR_SHURIKEN = (255, 255, 0)
    COLOR_ILLUSION = (100, 180, 255)
    COLOR_PARTICLE = (200, 200, 220)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_DETECTION_OFF = (60, 60, 80)
    COLOR_DETECTION_ON = (255, 0, 0)
    COLOR_GOAL = (0, 255, 150)

    # Physics
    GRAVITY = 0.8
    PLAYER_SPEED = 5
    PLAYER_JUMP_STRENGTH = -15
    PLAYER_FRICTION = -0.12
    MAX_FALL_SPEED = 12

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 18)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        
        # Game state variables
        self.player_pos = pygame.math.Vector2(0, 0)
        self.player_vel = pygame.math.Vector2(0, 0)
        self.player_rect = pygame.Rect(0, 0, 24, 40)
        self.on_ground = False
        self.is_stuck = False
        self.facing_right = True

        self.shurikens = []
        self.shuriken_cooldown = 0
        
        self.illusion_active = False
        self.illusion_pos = pygame.math.Vector2(0, 0)
        self.illusion_timer = 0
        self.illusion_cooldown = 0
        
        self.enemies = []
        
        self.particles = []
        
        self.platforms = []
        self.goal_rect = pygame.Rect(0, 0, 0, 0)

        self.camera_x = 0
        self.prev_distance_to_goal = 0

        self.prev_space_held = False
        self.prev_shift_held = False

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.detection_count = 0
        self.level_difficulty_modifier = 1.0

        # self.reset() is called by the wrapper
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.detection_count = 0

        # Player state
        self.player_pos = pygame.math.Vector2(100, 200)
        self.player_vel = pygame.math.Vector2(0, 0)
        self.is_stuck = False
        self.on_ground = False
        self.facing_right = True

        # Abilities state
        self.shurikens = []
        self.shuriken_cooldown = 0
        self.illusion_active = False
        self.illusion_timer = 0
        self.illusion_cooldown = 0

        self.particles = []
        
        # Level generation
        self._generate_level()

        self.prev_distance_to_goal = self._get_distance_to_goal()
        
        self.prev_space_held = False
        self.prev_shift_held = False

        return self._get_observation(), self._get_info()
    
    def _generate_level(self):
        self.platforms = [
            pygame.Rect(0, 360, self.WORLD_WIDTH, 40), # Floor
            pygame.Rect(0, 0, self.WORLD_WIDTH, 40),   # Ceiling
            pygame.Rect(300, 260, 150, 20),
            pygame.Rect(550, 180, 200, 20),
            pygame.Rect(800, 280, 100, 80),
            pygame.Rect(1000, 200, 20, 160),
            pygame.Rect(1100, 150, 250, 20),
            pygame.Rect(1450, 250, 150, 20),
            pygame.Rect(1700, 100, 20, 200),
            pygame.Rect(1700, 280, 200, 20),
            pygame.Rect(2000, 180, 300, 20),
        ]
        self.goal_rect = pygame.Rect(self.WORLD_WIDTH - 150, 260, 50, 100)
        
        self.enemies = [
            {'pos': pygame.math.Vector2(600, 156), 'patrol_start': 550, 'patrol_end': 750, 'dir': 1, 'state': 'patrol', 'stun_timer': 0},
            {'pos': pygame.math.Vector2(1200, 126), 'patrol_start': 1100, 'patrol_end': 1350, 'dir': 1, 'state': 'patrol', 'stun_timer': 0},
            {'pos': pygame.math.Vector2(1800, 256), 'patrol_start': 1700, 'patrol_end': 1900, 'dir': 1, 'state': 'patrol', 'stun_timer': 0},
            {'pos': pygame.math.Vector2(2100, 156), 'patrol_start': 2000, 'patrol_end': 2300, 'dir': 1, 'state': 'patrol', 'stun_timer': 0},
        ]

    def step(self, action):
        reward = 0
        terminated = False
        self.steps += 1
        
        # --- 1. HANDLE INPUT & COOLDOWNS ---
        movement = action[0]
        space_pressed = action[1] == 1 and not self.prev_space_held
        shift_pressed = action[2] == 1 and not self.prev_shift_held
        self.prev_space_held = action[1] == 1
        self.prev_shift_held = action[2] == 1

        if self.shuriken_cooldown > 0: self.shuriken_cooldown -= 1
        if self.illusion_cooldown > 0: self.illusion_cooldown -= 1
        if self.illusion_timer > 0: self.illusion_timer -= 1
        if self.illusion_timer == 0: self.illusion_active = False

        # --- 2. UPDATE PLAYER ---
        self._update_player(movement, space_pressed, shift_pressed)

        # --- 3. UPDATE GAME ENTITIES ---
        self._update_shurikens()
        self._update_enemies()
        self._update_particles()
        
        # --- 4. CHECK COLLISIONS & DETECTIONS ---
        detection_this_step, player_in_cone_but_safe = self._check_detections()
        if detection_this_step:
            self.detection_count += 1
            reward -= 50
            # sfx: detection_alert.wav
            self._create_particles(self.player_pos, 20, self.COLOR_DETECTION_ON, 5, 20)

        if player_in_cone_but_safe:
            reward -= 1

        stun_reward = self._check_shuriken_collisions()
        reward += stun_reward

        # --- 5. CALCULATE REWARDS & CHECK TERMINATION ---
        current_distance = self._get_distance_to_goal()
        reward += (self.prev_distance_to_goal - current_distance) * 0.01
        self.prev_distance_to_goal = current_distance

        if self.player_rect.colliderect(self.goal_rect):
            reward += 100
            terminated = True
            # sfx: level_complete.wav
        elif self.detection_count >= 3:
            reward -= 100
            terminated = True
            # sfx: game_over.wav
        elif self.steps >= self.MAX_STEPS:
            terminated = True
        
        self.score += reward
        self.game_over = terminated
        
        truncated = self.steps >= self.MAX_STEPS

        # --- 6. RETURN STATE ---
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _update_player(self, movement, space_pressed, shift_pressed):
        # --- Handle Actions ---
        if self.is_stuck:
            if movement in [1, 2, 3, 4]: # Any movement unsticks
                self.is_stuck = False
                # sfx: unstick.wav
                if movement == 1 and self.on_ground: # Jump from ground
                     self.player_vel.y = self.PLAYER_JUMP_STRENGTH
                     self._create_particles(self.player_pos + pygame.math.Vector2(0, self.player_rect.height/2), 10, self.COLOR_PARTICLE, 3, 15)
        else:
            # Horizontal Movement
            if movement == 3: # Left
                self.player_vel.x = -self.PLAYER_SPEED
                self.facing_right = False
            elif movement == 4: # Right
                self.player_vel.x = self.PLAYER_SPEED
                self.facing_right = True
            
            # Jumping
            if movement == 1 and self.on_ground:
                self.player_vel.y = self.PLAYER_JUMP_STRENGTH
                self.on_ground = False
                # sfx: jump.wav
                self._create_particles(self.player_pos + pygame.math.Vector2(0, self.player_rect.height/2), 10, self.COLOR_PARTICLE, 3, 15)

        # Throw Shuriken
        if space_pressed and self.shuriken_cooldown == 0:
            direction = 1 if self.facing_right else -1
            shuriken_vel = pygame.math.Vector2(15 * direction, 0)
            self.shurikens.append({'pos': self.player_pos.copy(), 'vel': shuriken_vel})
            self.shuriken_cooldown = 15 # 0.5s cooldown
            # sfx: shuriken_throw.wav

        # Activate Illusion
        if shift_pressed and self.illusion_cooldown == 0:
            self.illusion_active = True
            self.illusion_pos = self.player_pos.copy()
            self.illusion_timer = 90 # 3s duration
            self.illusion_cooldown = 300 # 10s cooldown
            # sfx: illusion_cast.wav
            self._create_particles(self.player_pos, 20, self.COLOR_ILLUSION, 4, 20)

        # --- Physics ---
        if not self.is_stuck:
            # Apply friction
            self.player_vel.x += self.player_vel.x * self.PLAYER_FRICTION
            if abs(self.player_vel.x) < 0.1: self.player_vel.x = 0
            
            # Apply gravity
            self.player_vel.y += self.GRAVITY
            if self.player_vel.y > self.MAX_FALL_SPEED:
                self.player_vel.y = self.MAX_FALL_SPEED

            # Move and collide
            self.player_pos.x += self.player_vel.x
            self.player_rect.centerx = int(self.player_pos.x)
            self._handle_collisions('horizontal')

            self.player_pos.y += self.player_vel.y
            self.player_rect.centery = int(self.player_pos.y)
            self.on_ground = False
            self._handle_collisions('vertical')
        else:
            self.player_vel = pygame.math.Vector2(0, 0)
        
        # Sticking logic
        if not self.is_stuck:
            if movement == 1: # Try stick to ceiling
                self.player_rect.y -= 1
                for plat in self.platforms:
                    if self.player_rect.colliderect(plat) and self.player_vel.y < 0:
                        self.is_stuck = True
                        self.player_pos.y = plat.bottom + self.player_rect.height / 2
                        # sfx: stick.wav
                        break
                self.player_rect.y += 1
            elif movement == 2 and self.on_ground: # Try stick to floor
                self.is_stuck = True
                # sfx: stick.wav

        # Clamp player to world
        self.player_pos.x = max(self.player_rect.width/2, min(self.player_pos.x, self.WORLD_WIDTH - self.player_rect.width/2))
        self.player_rect.center = self.player_pos

    def _handle_collisions(self, direction):
        for plat in self.platforms:
            if self.player_rect.colliderect(plat):
                if direction == 'horizontal':
                    if self.player_vel.x > 0: # Moving right
                        self.player_rect.right = plat.left
                    elif self.player_vel.x < 0: # Moving left
                        self.player_rect.left = plat.right
                    self.player_pos.x = self.player_rect.centerx
                    self.player_vel.x = 0
                elif direction == 'vertical':
                    if self.player_vel.y > 0: # Moving down
                        self.player_rect.bottom = plat.top
                        self.on_ground = True
                    elif self.player_vel.y < 0: # Moving up
                        self.player_rect.top = plat.bottom
                    self.player_pos.y = self.player_rect.centery
                    self.player_vel.y = 0

    def _update_shurikens(self):
        for s in self.shurikens[:]:
            s['pos'] += s['vel']
            if not (0 < s['pos'].x < self.WORLD_WIDTH):
                self.shurikens.remove(s)

    def _update_enemies(self):
        # Difficulty scaling
        speed_mod = 1.0 + (self.steps // 500) * 0.05
        
        for enemy in self.enemies:
            if enemy['state'] == 'stunned':
                enemy['stun_timer'] -= 1
                if enemy['stun_timer'] <= 0:
                    enemy['state'] = 'patrol'
                continue

            enemy['pos'].x += enemy['dir'] * 2 * speed_mod
            if enemy['pos'].x >= enemy['patrol_end']:
                enemy['dir'] = -1
            elif enemy['pos'].x <= enemy['patrol_start']:
                enemy['dir'] = 1

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _check_detections(self):
        detected = False
        in_cone_safe = False
        vision_range_mod = 1.0 + (self.steps // 1000) * 0.01

        for enemy in self.enemies:
            if enemy['state'] == 'stunned':
                continue

            vision_range = 150 * vision_range_mod
            p1 = enemy['pos']
            p2 = p1 + pygame.math.Vector2(enemy['dir'] * vision_range, -vision_range / 2)
            p3 = p1 + pygame.math.Vector2(enemy['dir'] * vision_range, vision_range / 2)
            vision_cone = [p1, p2, p3]

            if self._rect_in_polygon(self.player_rect, vision_cone):
                if self.illusion_active and self._rect_in_polygon(pygame.Rect(self.illusion_pos.x - 5, self.illusion_pos.y - 5, 10, 10), vision_cone):
                     in_cone_safe = True
                else:
                     detected = True
        return detected, in_cone_safe

    def _check_shuriken_collisions(self):
        reward = 0
        for s in self.shurikens[:]:
            shuriken_rect = pygame.Rect(s['pos'].x - 5, s['pos'].y - 5, 10, 10)
            for enemy in self.enemies:
                enemy_rect = pygame.Rect(enemy['pos'].x - 12, enemy['pos'].y - 20, 24, 40)
                if shuriken_rect.colliderect(enemy_rect):
                    if enemy['state'] != 'stunned':
                        enemy['state'] = 'stunned'
                        enemy['stun_timer'] = 120 # 4s stun
                        reward += 5
                        # sfx: enemy_stun.wav
                        self._create_particles(s['pos'], 15, self.COLOR_SHURIKEN, 4, 15)
                    if s in self.shurikens: self.shurikens.remove(s)
                    break
        return reward

    def _get_distance_to_goal(self):
        return abs(self.player_pos.x - self.goal_rect.centerx)

    def _get_observation(self):
        self._update_camera()
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _update_camera(self):
        target_camera_x = self.player_pos.x - self.SCREEN_WIDTH / 2
        # Smooth camera movement
        self.camera_x += (target_camera_x - self.camera_x) * 0.1
        self.camera_x = max(0, min(self.camera_x, self.WORLD_WIDTH - self.SCREEN_WIDTH))

    def _render_background(self):
        for i in range(0, self.WORLD_WIDTH, 50):
            x = int(i - self.camera_x)
            if 0 <= x < self.SCREEN_WIDTH:
                pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for i in range(0, self.SCREEN_HEIGHT, 50):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.SCREEN_WIDTH, i))

    def _render_game(self):
        # Goal
        goal_screen_rect = self.goal_rect.move(-self.camera_x, 0)
        pygame.draw.rect(self.screen, self.COLOR_GOAL, goal_screen_rect, 2)
        for i in range(goal_screen_rect.height // 20):
            pygame.draw.line(self.screen, self.COLOR_GOAL, 
                             (goal_screen_rect.left, goal_screen_rect.top + i*20), 
                             (goal_screen_rect.right, goal_screen_rect.top + i*20), 1)

        # Platforms
        for plat in self.platforms:
            screen_rect = plat.move(-self.camera_x, 0)
            if screen_rect.colliderect(self.screen.get_rect()):
                pygame.draw.rect(self.screen, self.COLOR_PLATFORM, screen_rect)
                top_rect = pygame.Rect(screen_rect.left, screen_rect.top, screen_rect.width, 4)
                pygame.draw.rect(self.screen, self.COLOR_PLATFORM_TOP, top_rect)
        
        # Particles
        for p in self.particles:
            pos = (int(p['pos'].x - self.camera_x), int(p['pos'].y))
            size = int(p['size'] * (p['life'] / p['max_life']))
            if size > 0:
                pygame.draw.circle(self.screen, p['color'], pos, size)

        # Enemies and vision cones
        for enemy in self.enemies:
            vision_range_mod = 1.0 + (self.steps // 1000) * 0.01
            vision_range = 150 * vision_range_mod
            p1 = enemy['pos']
            p2 = p1 + pygame.math.Vector2(enemy['dir'] * vision_range, -vision_range / 2)
            p3 = p1 + pygame.math.Vector2(enemy['dir'] * vision_range, vision_range / 2)
            
            vision_cone_world = [p1, p2, p3]
            vision_cone_screen = [(int(p.x - self.camera_x), int(p.y)) for p in vision_cone_world]
            
            if enemy['state'] != 'stunned':
                pygame.gfxdraw.aapolygon(self.screen, vision_cone_screen, self.COLOR_ENEMY_VISION)
                pygame.gfxdraw.filled_polygon(self.screen, vision_cone_screen, (*self.COLOR_ENEMY_VISION, 30))

            enemy_pos_screen = (int(enemy['pos'].x - self.camera_x), int(enemy['pos'].y))
            points = [(enemy_pos_screen[0], enemy_pos_screen[1] - 20), (enemy_pos_screen[0] - 12, enemy_pos_screen[1] + 20), (enemy_pos_screen[0] + 12, enemy_pos_screen[1] + 20)]
            pygame.draw.polygon(self.screen, self.COLOR_ENEMY, points)
            if enemy['state'] == 'stunned':
                # Stun effect
                angle = (self.steps % 30) * 12
                for i in range(3):
                    a = math.radians(angle + i * 120)
                    x = enemy_pos_screen[0] + math.cos(a) * 20
                    y = enemy_pos_screen[1] - 25 + math.sin(a) * 5
                    pygame.draw.circle(self.screen, self.COLOR_SHURIKEN, (int(x), int(y)), 2)


        # Shurikens
        for s in self.shurikens:
            pos = (int(s['pos'].x - self.camera_x), int(s['pos'].y))
            angle = (self.steps * 25) % 360
            self._draw_shuriken(pos, 8, angle)

        # Illusion
        if self.illusion_active:
            alpha = int(128 * (self.illusion_timer / 90.0))
            self._draw_player((self.illusion_pos.x - self.camera_x, self.illusion_pos.y), (*self.COLOR_ILLUSION, alpha), is_illusion=True)

        # Player
        player_screen_pos = (self.player_rect.centerx - self.camera_x, self.player_rect.centery)
        self._draw_player(player_screen_pos, self.COLOR_PLAYER)

    def _draw_player(self, pos, color, is_illusion=False):
        # Glow effect
        if not is_illusion:
            glow_color = (*self.COLOR_PLAYER_GLOW, 64) if not self.is_stuck else (*self.COLOR_SHURIKEN, 96)
            glow_surf = pygame.Surface((self.player_rect.width * 2, self.player_rect.height * 1.5), pygame.SRCALPHA)
            pygame.draw.ellipse(glow_surf, glow_color, glow_surf.get_rect())
            self.screen.blit(glow_surf, (int(pos[0] - self.player_rect.width), int(pos[1] - self.player_rect.height*0.75)))

        # Main body
        player_body_rect = pygame.Rect(0, 0, self.player_rect.width, self.player_rect.height)
        player_body_rect.center = pos
        
        if is_illusion:
            temp_surf = pygame.Surface(player_body_rect.size, pygame.SRCALPHA)
            temp_surf.fill((0,0,0,0))
            pygame.draw.rect(temp_surf, color, (0,0,player_body_rect.width, player_body_rect.height), border_radius=6)
            self.screen.blit(temp_surf, player_body_rect.topleft)
        else:
            pygame.draw.rect(self.screen, color, player_body_rect, border_radius=6)
            # Eye
            eye_x = pos[0] + (5 if self.facing_right else -5)
            eye_y = pos[1] - 8
            pygame.draw.rect(self.screen, self.COLOR_BG, (eye_x - 3, eye_y - 2, 6, 4))
            
    def _draw_shuriken(self, pos, size, angle):
        points = []
        for i in range(4):
            rad = math.radians(angle + i * 90)
            points.append((pos[0] + math.cos(rad) * size, pos[1] + math.sin(rad) * size))
            rad2 = math.radians(angle + 45 + i * 90)
            points.append((pos[0] + math.cos(rad2) * size/2, pos[1] + math.sin(rad2) * size/2))
        pygame.draw.polygon(self.screen, self.COLOR_SHURIKEN, points)
        pygame.draw.circle(self.screen, self.COLOR_SHURIKEN, pos, 2)


    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Detections
        detection_text = self.font_small.render("DETECTIONS:", True, self.COLOR_UI_TEXT)
        self.screen.blit(detection_text, (self.SCREEN_WIDTH - 150, 15))
        for i in range(3):
            color = self.COLOR_DETECTION_ON if i < self.detection_count else self.COLOR_DETECTION_OFF
            pygame.draw.circle(self.screen, color, (self.SCREEN_WIDTH - 40 + i * 20, 23), 6)

        # Cooldowns
        if self.shuriken_cooldown > 0:
            pygame.draw.rect(self.screen, self.COLOR_SHURIKEN, (10, 40, 100 * (self.shuriken_cooldown / 15), 5))
        if self.illusion_cooldown > 0:
            pygame.draw.rect(self.screen, self.COLOR_ILLUSION, (10, 50, 100 * (self.illusion_cooldown / 300), 5))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "detections": self.detection_count,
            "player_pos": (self.player_pos.x, self.player_pos.y)
        }
        
    def _create_particles(self, pos, count, color, speed, life):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * random.uniform(1, speed)
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'life': random.randint(life // 2, life),
                'max_life': life,
                'color': color,
                'size': random.uniform(1, 4)
            })

    @staticmethod
    def _rect_in_polygon(rect, polygon):
        # Simple check if any corner of the rect is inside the polygon
        # This is an approximation but sufficient for this game's purpose
        points = [rect.topleft, rect.topright, rect.bottomleft, rect.bottomright]
        for p in points:
            if GameEnv._point_in_polygon(p, polygon):
                return True
        return False

    @staticmethod
    def _point_in_polygon(point, polygon):
        x, y = point
        n = len(polygon)
        inside = False
        p1x, p1y = polygon[0]
        for i in range(n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside
    
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # This method is for self-testing and not required by Gymnasium.
        # It's good practice to keep it for development.
        print("Running implementation validation...")
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
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example usage to test the environment
if __name__ == '__main__':
    # This block will not run in the hosted environment but is useful for local testing.
    # Un-comment the line below to run with a visible display.
    os.environ["SDL_VIDEODRIVER"] = "x11" 

    env = GameEnv(render_mode="rgb_array")
    env.reset()
    env.validate_implementation()
    
    # --- Manual Play Loop ---
    obs, info = env.reset()
    done = False
    
    # Use a separate display for human play
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Magnetic Ninja")
    
    # Action state
    movement_action = 0 # 0: none
    space_action = 0 # 0: released
    shift_action = 0 # 0: released
    
    clock = pygame.time.Clock()
    
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            
            # Key presses
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE: space_action = 1
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: shift_action = 1
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_SPACE: space_action = 0
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: shift_action = 0
                
        # Continuous key holds for movement
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement_action = 1
        elif keys[pygame.K_DOWN]: movement_action = 2
        elif keys[pygame.K_LEFT]: movement_action = 3
        elif keys[pygame.K_RIGHT]: movement_action = 4
        else: movement_action = 0

        action = [movement_action, space_action, shift_action]
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Update the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(GameEnv.FPS)

    print(f"Game Over! Final Score: {info['score']}, Steps: {info['steps']}")
    env.close()