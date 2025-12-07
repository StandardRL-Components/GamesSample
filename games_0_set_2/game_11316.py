import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:47:34.188344
# Source Brief: brief_01316.md
# Brief Index: 1316
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Gymnasium environment for "Gravity Flip Pyramid".

    The agent controls a character that can move and flip gravity.
    The goal is to navigate a room with patrolling enemies, solve a simple
    puzzle by activating a switch to open a door, and reach the exit.
    Enemies have a two-stage vision cone: a warning zone and an instant-fail
    detection zone. Visual polish, smooth animations, and particle effects
    are prioritized to create a high-quality gameplay experience.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Navigate a treacherous pyramid by flipping gravity to your advantage. "
        "Avoid patrolling guards, activate the switch, and escape through the exit."
    )
    user_guide = (
        "Use the arrow keys (↑↓←→) to move your character. "
        "Press space to flip gravity and navigate around obstacles and enemies."
    )
    auto_advance = True

    # --- CONSTANTS ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    MAX_STEPS = 5000

    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_WALL = (40, 50, 70)
    COLOR_PLAYER = (100, 200, 255)
    COLOR_PLAYER_GLOW = (100, 200, 255, 50)
    COLOR_ENEMY = (255, 50, 50)
    COLOR_VISION_WARN = (255, 200, 0, 70)
    COLOR_VISION_DETECT = (255, 50, 50, 100)
    COLOR_EXIT = (50, 255, 50)
    COLOR_SWITCH_OFF = (255, 150, 0)
    COLOR_SWITCH_ON = (50, 255, 200)
    COLOR_DOOR = (200, 100, 50)
    COLOR_TEXT = (230, 230, 240)
    COLOR_TRAP_LASER = (255, 100, 0)

    # Game Mechanics
    GRAVITY_FLIP_COOLDOWN_FRAMES = 30
    ENEMY_BASE_SPEED = 1.0
    ENEMY_SPEED_INCREASE_PER_500_STEPS = 0.05
    ENEMY_VISION_WARN_RANGE = 150
    ENEMY_VISION_DETECT_RANGE = 75
    ENEMY_VISION_ANGLE = 45 # Degrees

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 50, bold=True)
        
        # --- State Variables (initialized in reset) ---
        self.steps = None
        self.score = None
        self.game_over = None
        self.trap_triggered_by = None
        self.reward_this_step = None
        self.gravity_up = None
        self.gravity_flip_cooldown = None
        self.player_pos = None
        self.player_size = None
        self.player_speed = None
        self.walls = None
        self.exit_rect = None
        self.switch_rect = None
        self.switch_active = None
        self.door_rect = None
        self.enemies = None
        self.was_in_warning_zone = None
        self.particles = None
        
        # self.reset() # This is called by the wrapper, no need to call it here.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.trap_triggered_by = None
        self.reward_this_step = 0.0

        # Gravity and Player
        self.gravity_up = False
        self.gravity_flip_cooldown = 0
        self.player_pos = pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 50)
        self._update_player_properties()

        # Level Layout
        self.walls = [
            pygame.Rect(0, 0, self.SCREEN_WIDTH, 20),
            pygame.Rect(0, self.SCREEN_HEIGHT - 20, self.SCREEN_WIDTH, 20),
            pygame.Rect(0, 0, 20, self.SCREEN_HEIGHT),
            pygame.Rect(self.SCREEN_WIDTH - 20, 0, 20, self.SCREEN_HEIGHT),
        ]
        self.exit_rect = pygame.Rect(self.SCREEN_WIDTH / 2 - 25, 20, 50, 10)
        self.switch_rect = pygame.Rect(100, 180, 20, 40)
        self.switch_active = False
        self.door_rect = pygame.Rect(self.SCREEN_WIDTH - 120, 150, 20, 100)
        
        # Enemies
        self._spawn_enemies()
        self.was_in_warning_zone = False

        # Effects
        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()

        self.reward_this_step = 0.0
        self.steps += 1
        
        self._handle_input(action)
        self._update_player_movement(action[0])
        self._update_enemies()
        self._check_interactions()
        self._update_particles()
        
        if not self.was_in_warning_zone:
            self.reward_this_step += 0.1

        self.score += self.reward_this_step
        terminated = self.game_over
        truncated = self.steps >= self.MAX_STEPS

        return (
            self._get_observation(),
            self.reward_this_step,
            terminated,
            truncated,
            self._get_info(),
        )

    def _handle_input(self, action):
        space_pressed = action[1] == 1
        
        if space_pressed and self.gravity_flip_cooldown <= 0:
            self.gravity_up = not self.gravity_up
            self.gravity_flip_cooldown = self.GRAVITY_FLIP_COOLDOWN_FRAMES
            self._update_player_properties()
            # Sound: Gravity_Flip.wav
            self._create_gravity_flip_effect()

        if self.gravity_flip_cooldown > 0:
            self.gravity_flip_cooldown -= 1

    def _update_player_movement(self, movement_action):
        move_vec = pygame.Vector2(0, 0)
        
        # Map actions to movement vectors based on gravity
        if movement_action == 1:  # Up
            move_vec.y = -1 if not self.gravity_up else 1
        elif movement_action == 2:  # Down
            move_vec.y = 1 if not self.gravity_up else -1
        elif movement_action == 3:  # Left
            move_vec.x = -1
        elif movement_action == 4:  # Right
            move_vec.x = 1

        if move_vec.length() > 0:
            move_vec.scale_to_length(self.player_speed)
            # Create movement particles
            if self.steps % 3 == 0:
                self._create_particle(self.player_pos, -move_vec * 0.2, 10, self.COLOR_PLAYER, 3)

        # Move and check for wall collisions
        new_pos = self.player_pos + move_vec
        player_rect = pygame.Rect(new_pos.x - self.player_size, new_pos.y - self.player_size, self.player_size * 2, self.player_size * 2)

        collided = False
        for wall in self.walls:
            if wall.colliderect(player_rect):
                collided = True
                break
        if not self.switch_active and self.door_rect.colliderect(player_rect):
            collided = True

        if not collided:
            self.player_pos = new_pos
            
    def _update_enemies(self):
        current_speed = self.ENEMY_BASE_SPEED + self.ENEMY_SPEED_INCREASE_PER_500_STEPS * (self.steps // 500)
        for enemy in self.enemies:
            target_pos = enemy['path'][enemy['target_idx']]
            direction = target_pos - enemy['pos']
            
            if direction.length() < current_speed:
                enemy['pos'] = target_pos
                enemy['target_idx'] = 1 - enemy['target_idx'] # Flip between 0 and 1
            else:
                enemy['pos'] += direction.normalize() * current_speed

    def _check_interactions(self):
        player_rect = pygame.Rect(self.player_pos.x - self.player_size, self.player_pos.y - self.player_size, self.player_size * 2, self.player_size * 2)

        # Exit condition
        if self.exit_rect.colliderect(player_rect):
            self.reward_this_step += 100
            self.game_over = True
            # Sound: Win.wav
            return

        # Switch interaction
        if self.switch_rect.colliderect(player_rect) and self.gravity_flip_cooldown > 0: # Requires a flip to activate
            if not self.switch_active:
                self.switch_active = True
                self.reward_this_step += 5
                # Sound: Switch_Activate.wav
                self._create_burst(pygame.Vector2(self.switch_rect.center), self.COLOR_SWITCH_ON)

        # Enemy detection
        is_in_warning_zone = False
        for enemy in self.enemies:
            # Detection Zone (Fatal)
            if self._is_point_in_cone(self.player_pos, enemy['pos'], enemy['path'][enemy['target_idx']] - enemy['pos'], self.ENEMY_VISION_DETECT_RANGE, self.ENEMY_VISION_ANGLE):
                self.reward_this_step = -100 # Override all other rewards
                self.game_over = True
                self.trap_triggered_by = enemy
                # Sound: Trap_Activate.wav
                return
            
            # Warning Zone
            if self._is_point_in_cone(self.player_pos, enemy['pos'], enemy['path'][enemy['target_idx']] - enemy['pos'], self.ENEMY_VISION_WARN_RANGE, self.ENEMY_VISION_ANGLE):
                is_in_warning_zone = True
        
        if is_in_warning_zone and not self.was_in_warning_zone:
            self.reward_this_step -= 1
            # Sound: Detected_Warning.wav
        
        self.was_in_warning_zone = is_in_warning_zone
        
    def _update_particles(self):
        self.particles = [p for p in self.particles if p['lifespan'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            p['size'] = max(0, p['size'] * 0.95)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    # --- RENDER METHODS ---
    def _render_game(self):
        # Walls and Level
        for wall in self.walls:
            pygame.draw.rect(self.screen, self.COLOR_WALL, wall)
        pygame.draw.rect(self.screen, self.COLOR_EXIT, self.exit_rect)
        switch_color = self.COLOR_SWITCH_ON if self.switch_active else self.COLOR_SWITCH_OFF
        pygame.draw.rect(self.screen, switch_color, self.switch_rect)
        if not self.switch_active:
            pygame.draw.rect(self.screen, self.COLOR_DOOR, self.door_rect)

        # Particles
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], p['pos'], p['size'])

        # Enemies and Vision Cones
        for enemy in self.enemies:
            # Vision Cones (draw first to be underneath)
            direction_vec = (enemy['path'][enemy['target_idx']] - enemy['pos']).normalize()
            self._draw_vision_cone(enemy['pos'], direction_vec, self.ENEMY_VISION_WARN_RANGE, self.ENEMY_VISION_ANGLE, self.COLOR_VISION_WARN)
            self._draw_vision_cone(enemy['pos'], direction_vec, self.ENEMY_VISION_DETECT_RANGE, self.ENEMY_VISION_ANGLE, self.COLOR_VISION_DETECT)
            # Enemy Body
            pygame.gfxdraw.aacircle(self.screen, int(enemy['pos'].x), int(enemy['pos'].y), 12, self.COLOR_ENEMY)
            pygame.gfxdraw.filled_circle(self.screen, int(enemy['pos'].x), int(enemy['pos'].y), 12, self.COLOR_ENEMY)

        # Player
        self._draw_glow_circle(self.player_pos, self.player_size, self.COLOR_PLAYER, self.COLOR_PLAYER_GLOW)
        
        # Trap effect on game over
        if self.game_over and self.trap_triggered_by:
            pygame.draw.line(self.screen, self.COLOR_TRAP_LASER, self.trap_triggered_by['pos'], self.player_pos, 5)
            pygame.draw.line(self.screen, (255,255,255), self.trap_triggered_by['pos'], self.player_pos, 2)


    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {int(self.score):04d}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (30, self.SCREEN_HEIGHT - 35))

        # Gravity Indicator
        arrow_points = []
        if self.gravity_up:
            arrow_points = [(self.SCREEN_WIDTH - 40, 30), (self.SCREEN_WIDTH - 50, 45), (self.SCREEN_WIDTH - 30, 45)]
        else:
            arrow_points = [(self.SCREEN_WIDTH - 40, 45), (self.SCREEN_WIDTH - 50, 30), (self.SCREEN_WIDTH - 30, 30)]
        pygame.draw.polygon(self.screen, self.COLOR_TEXT, arrow_points)
        
        # Game Over Text
        if self.game_over:
            msg = "VICTORY!" if self.reward_this_step > 0 else "TRAP ACTIVATED"
            color = self.COLOR_EXIT if self.reward_this_step > 0 else self.COLOR_ENEMY
            end_text = self.font_game_over.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            s = pygame.Surface(text_rect.inflate(20, 20).size, pygame.SRCALPHA)
            s.fill((0,0,0,180))
            self.screen.blit(s, text_rect.inflate(20,20).topleft)
            self.screen.blit(end_text, text_rect)

    # --- HELPER METHODS ---
    def _update_player_properties(self):
        if self.gravity_up:
            self.player_size = 15
            self.player_speed = 2.5
        else:
            self.player_size = 10
            self.player_speed = 4.0

    def _spawn_enemies(self):
        self.enemies = [
            {'pos': pygame.Vector2(150, 100), 'path': [pygame.Vector2(150, 100), pygame.Vector2(450, 100)], 'target_idx': 1},
            {'pos': pygame.Vector2(500, 300), 'path': [pygame.Vector2(500, 300), pygame.Vector2(100, 300)], 'target_idx': 1}
        ]
        
    def _is_point_in_cone(self, point, cone_apex, cone_direction, cone_range, cone_angle_deg):
        if cone_direction.length() == 0: return False
        
        vec_to_point = point - cone_apex
        dist_to_point = vec_to_point.length()

        if dist_to_point > cone_range or dist_to_point == 0:
            return False

        angle_rad = math.radians(cone_angle_deg / 2)
        cone_dir_norm = cone_direction.normalize()
        vec_to_point_norm = vec_to_point.normalize()
        
        angle_between = math.acos(cone_dir_norm.dot(vec_to_point_norm))
        
        return angle_between < angle_rad

    def _draw_vision_cone(self, apex, direction, range, angle_deg, color):
        if direction.length() == 0: return
        
        angle_rad = math.radians(angle_deg / 2)
        
        # Calculate the two edge vectors of the cone
        left_vec = direction.rotate_rad(-angle_rad)
        right_vec = direction.rotate_rad(angle_rad)
        
        p1 = apex
        p2 = apex + left_vec.normalize() * range
        p3 = apex + right_vec.normalize() * range
        
        pygame.gfxdraw.aapolygon(self.screen, [(p1.x, p1.y), (p2.x, p2.y), (p3.x, p3.y)], color)
        pygame.gfxdraw.filled_polygon(self.screen, [(p1.x, p1.y), (p2.x, p2.y), (p3.x, p3.y)], color)

    def _draw_glow_circle(self, pos, radius, color, glow_color):
        for i in range(4, 0, -1):
            alpha_glow_color = (glow_color[0], glow_color[1], glow_color[2], glow_color[3] // i)
            pygame.gfxdraw.aacircle(self.screen, int(pos.x), int(pos.y), int(radius + i * 2), alpha_glow_color)
            pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), int(radius + i * 2), alpha_glow_color)
        
        pygame.gfxdraw.aacircle(self.screen, int(pos.x), int(pos.y), int(radius), color)
        pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), int(radius), color)

    def _create_particle(self, pos, vel, lifespan, color, size):
        self.particles.append({'pos': pos.copy(), 'vel': vel.copy(), 'lifespan': lifespan, 'color': color, 'size': size})

    def _create_burst(self, pos, color):
        for _ in range(30):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            lifespan = self.np_random.integers(15, 31)
            size = self.np_random.uniform(2, 5)
            self._create_particle(pos, vel, lifespan, color, size)

    def _create_gravity_flip_effect(self):
        for i in range(self.SCREEN_WIDTH // 20):
            pos = pygame.Vector2(i * 20, self.SCREEN_HEIGHT / 2)
            vel = pygame.Vector2(0, 3 if self.gravity_up else -3)
            self._create_particle(pos, vel, 20, (200, 200, 255, 100), 4)

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block is for manual play testing.
    # It will not be executed by the test suite.
    # You can make changes here to test your environment.
    
    # Un-comment the following line to run in a window
    os.environ.pop("SDL_VIDEODRIVER", None)

    env = GameEnv()
    obs, info = env.reset(seed=42)
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Gravity Flip Pyramid")
    clock = pygame.time.Clock()

    running = True
    total_reward = 0
    while running:
        # --- Action Mapping for Human Player ---
        keys = pygame.key.get_pressed()
        movement = 0 # No-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement, space_held, shift_held]

        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Pygame Rendering ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                total_reward = 0

        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward}")
            pygame.time.wait(2000) # Pause for 2 seconds
            obs, info = env.reset()
            total_reward = 0

        clock.tick(env.FPS)
    
    env.close()