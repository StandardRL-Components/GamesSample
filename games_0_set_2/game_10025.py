import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T09:42:27.992496
# Source Brief: brief_00025.md
# Brief Index: 25
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A retro-styled platformer where a spy uses a magnetic grappling gun and darts
    to infiltrate enemy facilities and complete missions.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Infiltrate enemy facilities as a spy using a magnetic grappling gun and disabling darts to complete your mission."
    )
    user_guide = (
        "Controls: ←→ to move, ↑ to jump. Press space to fire a dart. Press and hold shift to use the grappling hook."
    )
    auto_advance = True

    # --- Constants ---
    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_PLATFORM = (60, 70, 90)
    COLOR_PLAYER = (50, 255, 255)
    COLOR_PLAYER_GLOW = (50, 255, 255, 50)
    COLOR_GUARD = (255, 80, 80)
    COLOR_GUARD_VISION = (255, 80, 80, 40)
    COLOR_OBJECTIVE = (80, 255, 80)
    COLOR_DART = (255, 255, 100)
    COLOR_GRAPPLE_CABLE = (200, 200, 200)
    COLOR_GRAPPLE_HOOK = (220, 220, 250)
    COLOR_SEC_SYS_ACTIVE = (255, 180, 50)
    COLOR_SEC_SYS_INACTIVE = (100, 100, 120)
    COLOR_TRAP = (255, 0, 0)
    COLOR_TEXT = (240, 240, 240)
    COLOR_DOOR = (255, 180, 50, 150)

    # Physics
    GRAVITY = 0.5
    FRICTION = -0.12
    PLAYER_ACCEL = 0.8
    PLAYER_JUMP_STRENGTH = -10
    PLAYER_MAX_SPEED = 6
    DART_SPEED = 15
    GRAPPLE_SPEED = 20
    GRAPPLE_PULL_SPEED = 7

    # Game settings
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 2000
    DART_COOLDOWN_FRAMES = 15
    INITIAL_DARTS = 10

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        self.render_mode = render_mode

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)

        # Initialize state variables (will be properly set in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = pygame.Vector2(0, 0)
        self.player_vel = pygame.Vector2(0, 0)
        self.on_ground = False
        self.last_move_dir = pygame.Vector2(1, 0)
        self.remaining_darts = 0
        self.dart_cooldown = 0
        self.darts = []
        self.guards = []
        self.security_systems = []
        self.objective_pos = pygame.Vector2(0, 0)
        self.platforms = []
        self.traps = []
        self.door_rect = None
        self.particles = []
        self.grapple_state = "inactive" # inactive, extending, attached, retracting
        self.grapple_pos = pygame.Vector2(0, 0)
        self.grapple_vel = pygame.Vector2(0, 0)
        self.grapple_attach_point = pygame.Vector2(0, 0)
        self.prev_space_held = False
        self.prev_shift_held = False
        self.last_dist_to_objective = 0
        self.last_reward_info = {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False

        # Level layout
        self.platforms = [
            pygame.Rect(0, 380, 640, 20),      # Floor
            pygame.Rect(100, 300, 150, 20),
            pygame.Rect(350, 250, 150, 20),
            pygame.Rect(0, 180, 120, 20),
            pygame.Rect(200, 100, 240, 20),
        ]
        self.traps = [pygame.Rect(250, 370, 100, 10)]

        # Player
        self.player_pos = pygame.Vector2(50, 350)
        self.player_vel = pygame.Vector2(0, 0)
        self.on_ground = False
        self.last_move_dir = pygame.Vector2(1, 0)

        # Gadgets
        self.remaining_darts = self.INITIAL_DARTS
        self.dart_cooldown = 0
        self.darts = []
        self.grapple_state = "inactive"

        # Enemies and objectives
        self.guards = [{
            'rect': pygame.Rect(360, 220, 20, 30),
            'path_start': 360, 'path_end': 480, 'speed': 1,
            'state': 'patrolling', 'direction': 1, 'vision_angle': 45
        }]
        self.security_systems = [{'rect': pygame.Rect(290, 80, 20, 20), 'active': True}]
        self.door_rect = pygame.Rect(580, 300, 20, 80)
        self.objective_pos = pygame.Vector2(610, 350)

        self.particles = []
        self.prev_space_held = False
        self.prev_shift_held = False
        self.last_dist_to_objective = self.player_pos.distance_to(self.objective_pos)
        self.last_reward_info = {}

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.last_reward_info = {} # Reset event reward tracker

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.prev_space_held
        shift_pressed = shift_held and not self.prev_shift_held
        shift_released = not shift_held and self.prev_shift_held

        self._handle_input(movement, space_pressed, shift_pressed, shift_released)
        self._update_player()
        self._update_grapple()
        self._update_darts()
        self._update_guards()
        self._update_particles()

        reward = self._calculate_reward()
        self.score += reward
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_input(self, movement, space_pressed, shift_pressed, shift_released):
        # Movement
        current_move_dir = pygame.Vector2(0, 0)
        if movement == 3: # Left
            self.player_vel.x -= self.PLAYER_ACCEL
            current_move_dir.x = -1
        if movement == 4: # Right
            self.player_vel.x += self.PLAYER_ACCEL
            current_move_dir.x = 1
        if movement == 1 and self.on_ground: # Up (Jump)
            self.player_vel.y = self.PLAYER_JUMP_STRENGTH
            self._create_particles(self.player_pos + pygame.Vector2(0, 10), 5, (200,200,200))
            # sfx: jump

        if current_move_dir.length_squared() > 0:
            self.last_move_dir = current_move_dir.normalize()

        # Fire Dart
        if space_pressed and self.dart_cooldown == 0 and self.remaining_darts > 0:
            self.remaining_darts -= 1
            self.dart_cooldown = self.DART_COOLDOWN_FRAMES
            dart_vel = self.last_move_dir.normalize() * self.DART_SPEED if self.last_move_dir.length() > 0 else pygame.Vector2(1, 0) * self.DART_SPEED
            self.darts.append({'pos': self.player_pos.copy(), 'vel': dart_vel})
            # sfx: dart_fire

        # Fire Grapple
        if shift_pressed and self.grapple_state == "inactive":
            self.grapple_state = "extending"
            self.grapple_pos = self.player_pos.copy()
            self.grapple_vel = self.last_move_dir.normalize() * self.GRAPPLE_SPEED if self.last_move_dir.length() > 0 else pygame.Vector2(0, -1) * self.GRAPPLE_SPEED
            # sfx: grapple_fire

        # Release Grapple
        if shift_released and self.grapple_state in ["attached", "extending"]:
            self.grapple_state = "inactive"

    def _update_player(self):
        # Apply physics if not grappling
        if self.grapple_state != "attached":
            # Gravity
            self.player_vel.y += self.GRAVITY
            # Friction
            self.player_vel.x += self.player_vel.x * self.FRICTION
            if abs(self.player_vel.x) < 0.1: self.player_vel.x = 0
            # Cap speed
            self.player_vel.x = max(-self.PLAYER_MAX_SPEED, min(self.PLAYER_MAX_SPEED, self.player_vel.x))
        else: # Is grappling
            pull_dir = (self.grapple_attach_point - self.player_pos).normalize()
            self.player_vel = pull_dir * self.GRAPPLE_PULL_SPEED
            if self.player_pos.distance_to(self.grapple_attach_point) < 10:
                self.grapple_state = "inactive" # Arrived at destination
                self.player_vel = pygame.Vector2(0, 0)

        # Move and collide
        self.player_pos.x += self.player_vel.x
        player_rect = pygame.Rect(self.player_pos.x - 10, self.player_pos.y - 20, 20, 30)
        self._collide_player(player_rect, 'x')

        self.player_pos.y += self.player_vel.y
        player_rect.topleft = (self.player_pos.x - 10, self.player_pos.y - 20)
        self.on_ground = False
        self._collide_player(player_rect, 'y')

        # Dart cooldown
        if self.dart_cooldown > 0:
            self.dart_cooldown -= 1

        # Boundaries
        self.player_pos.x = max(10, min(self.SCREEN_WIDTH - 10, self.player_pos.x))
        if self.player_pos.y > self.SCREEN_HEIGHT + 20: # Fell off screen
             self.game_over = True

    def _collide_player(self, player_rect, axis):
        collidables = self.platforms.copy()
        if self.security_systems[0]['active']:
            collidables.append(self.door_rect)

        for rect in collidables:
            if player_rect.colliderect(rect):
                if axis == 'x':
                    if self.player_vel.x > 0: player_rect.right = rect.left
                    if self.player_vel.x < 0: player_rect.left = rect.right
                    self.player_pos.x = player_rect.centerx
                    self.player_vel.x = 0
                elif axis == 'y':
                    if self.player_vel.y > 0:
                        player_rect.bottom = rect.top
                        self.on_ground = True
                    if self.player_vel.y < 0:
                        player_rect.top = rect.bottom
                    self.player_pos.y = player_rect.centery + 10
                    self.player_vel.y = 0

    def _update_grapple(self):
        if self.grapple_state == "extending":
            self.grapple_pos += self.grapple_vel
            grapple_rect = pygame.Rect(self.grapple_pos.x - 2, self.grapple_pos.y - 2, 4, 4)

            # Check collision with platforms
            for plat in self.platforms:
                if plat.colliderect(grapple_rect):
                    self.grapple_state = "attached"
                    self.grapple_attach_point = self.grapple_pos.copy()
                    # sfx: grapple_attach
                    return

            # Check screen bounds
            if not self.screen.get_rect().contains(grapple_rect):
                self.grapple_state = "inactive"

    def _update_darts(self):
        darts_to_remove = []
        for i, dart in enumerate(self.darts):
            dart['pos'] += dart['vel']
            dart_rect = pygame.Rect(dart['pos'].x - 3, dart['pos'].y - 3, 6, 6)
            
            # Hit guard
            for guard in self.guards:
                if guard['rect'].colliderect(dart_rect) and guard['state'] != 'knocked_out':
                    guard['state'] = 'knocked_out'
                    self.last_reward_info['knockout'] = True
                    self._create_particles(dart['pos'], 15, self.COLOR_GUARD)
                    # sfx: guard_hit
                    if i not in darts_to_remove: darts_to_remove.append(i)
                    continue
            
            # Hit security system
            for sys in self.security_systems:
                if sys['rect'].colliderect(dart_rect) and sys['active']:
                    sys['active'] = False
                    self.last_reward_info['disable_system'] = True
                    self._create_particles(dart['pos'], 20, self.COLOR_SEC_SYS_ACTIVE)
                    # sfx: system_disabled
                    if i not in darts_to_remove: darts_to_remove.append(i)
                    continue
            
            # Hit wall
            for plat in self.platforms:
                if plat.colliderect(dart_rect):
                    self._create_particles(dart['pos'], 5, self.COLOR_DART)
                    if i not in darts_to_remove: darts_to_remove.append(i)
                    continue

            # Out of bounds
            if not self.screen.get_rect().contains(dart_rect):
                if i not in darts_to_remove: darts_to_remove.append(i)
        
        # Remove used darts
        for i in sorted(darts_to_remove, reverse=True):
            del self.darts[i]

    def _update_guards(self):
        player_rect = pygame.Rect(self.player_pos.x - 10, self.player_pos.y - 20, 20, 30)
        
        # Difficulty scaling
        patrol_speed_increase = (self.steps // 500) * 0.05
        vision_angle_increase = (self.steps // 1000) * 1

        for guard in self.guards:
            if guard['state'] == 'patrolling':
                guard['rect'].x += guard['speed'] * guard['direction'] * (1 + patrol_speed_increase)
                if guard['rect'].right > guard['path_end']: guard['direction'] = -1
                if guard['rect'].left < guard['path_start']: guard['direction'] = 1

                # Vision check
                dist_to_player = self.player_pos.distance_to(guard['rect'].center)
                if dist_to_player < 150:
                    vec_to_player = self.player_pos - pygame.Vector2(guard['rect'].center)
                    guard_facing_dir = pygame.Vector2(guard['direction'], 0)
                    angle_to_player = guard_facing_dir.angle_to(vec_to_player)

                    if abs(angle_to_player) < guard['vision_angle'] + vision_angle_increase:
                        # Line of sight check
                        can_see = True
                        for plat in self.platforms:
                            if plat.clipline(guard['rect'].center, self.player_pos):
                                can_see = False
                                break
                        if can_see:
                            guard['state'] = 'alerted'
                            self.game_over = True
                            self.last_reward_info['detected'] = True
                            # sfx: detected
            elif guard['state'] == 'knocked_out':
                pass # Stays knocked out

    def _update_particles(self):
        particles_to_remove = []
        for i, p in enumerate(self.particles):
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                particles_to_remove.append(i)
        for i in sorted(particles_to_remove, reverse=True):
            del self.particles[i]

    def _calculate_reward(self):
        reward = 0
        
        # Distance to objective
        dist = self.player_pos.distance_to(self.objective_pos)
        reward += (self.last_dist_to_objective - dist) * 0.1
        self.last_dist_to_objective = dist

        # Event rewards
        if self.last_reward_info.get('knockout'):
            reward += 10.0
        if self.last_reward_info.get('disable_system'):
            reward += 5.0
        if self.last_reward_info.get('detected'):
            reward -= 5.0

        # Terminal rewards
        player_rect = pygame.Rect(self.player_pos.x - 10, self.player_pos.y - 20, 20, 30)
        if player_rect.colliderect(pygame.Rect(self.objective_pos.x - 10, self.objective_pos.y - 10, 20, 20)):
            reward += 100.0

        return reward

    def _check_termination(self):
        if self.game_over: # Already set by detection, falling, etc.
            return True
        
        player_rect = pygame.Rect(self.player_pos.x - 10, self.player_pos.y - 20, 20, 30)
        
        # Reached objective
        if player_rect.colliderect(pygame.Rect(self.objective_pos.x - 10, self.objective_pos.y - 10, 20, 20)):
            self.game_over = True
            return True

        # Hit a trap
        for trap in self.traps:
            if player_rect.colliderect(trap):
                self.game_over = True
                return True
        
        # Max steps is handled by truncation
        
        # Softlock: out of darts and system is still active
        if self.remaining_darts <= 0 and self.security_systems[0]['active']:
            self.game_over = True
            return True

        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "darts": self.remaining_darts}

    def _render_game(self):
        # Platforms
        for plat in self.platforms:
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, plat)
        # Traps
        for trap in self.traps:
            pygame.draw.rect(self.screen, self.COLOR_TRAP, trap)
        # Objective
        pygame.gfxdraw.filled_circle(self.screen, int(self.objective_pos.x), int(self.objective_pos.y), 10, self.COLOR_OBJECTIVE)
        pygame.gfxdraw.aacircle(self.screen, int(self.objective_pos.x), int(self.objective_pos.y), 10, self.COLOR_OBJECTIVE)
        # Security System & Door
        for sys in self.security_systems:
            color = self.COLOR_SEC_SYS_ACTIVE if sys['active'] else self.COLOR_SEC_SYS_INACTIVE
            pygame.draw.rect(self.screen, color, sys['rect'])
            if sys['active']:
                pygame.draw.rect(self.screen, self.COLOR_DOOR, self.door_rect)

        # Guards
        for guard in self.guards:
            if guard['state'] != 'knocked_out':
                pygame.draw.rect(self.screen, self.COLOR_GUARD, guard['rect'])
                # Vision Cone
                p1 = guard['rect'].center
                vision_angle = guard['vision_angle'] + (self.steps // 1000)
                p2_angle = math.radians(-vision_angle)
                p3_angle = math.radians(vision_angle)
                if guard['direction'] == -1:
                    p2_angle += math.pi
                    p3_angle += math.pi
                
                p2 = (p1[0] + 150 * math.cos(p2_angle), p1[1] + 150 * math.sin(p2_angle))
                p3 = (p1[0] + 150 * math.cos(p3_angle), p1[1] + 150 * math.sin(p3_angle))
                pygame.gfxdraw.aapolygon(self.screen, [p1, p2, p3], self.COLOR_GUARD_VISION)
                pygame.gfxdraw.filled_polygon(self.screen, [p1, p2, p3], self.COLOR_GUARD_VISION)
            else: # Knocked out
                r = guard['rect']
                pygame.draw.rect(self.screen, (100,50,50), pygame.Rect(r.x, r.bottom-10, r.width, 10))


        # Grapple
        if self.grapple_state in ["extending", "attached"]:
            pygame.gfxdraw.line(self.screen, int(self.player_pos.x), int(self.player_pos.y-10), int(self.grapple_pos.x), int(self.grapple_pos.y), self.COLOR_GRAPPLE_CABLE)
            pygame.gfxdraw.filled_circle(self.screen, int(self.grapple_pos.x), int(self.grapple_pos.y), 4, self.COLOR_GRAPPLE_HOOK)

        # Player
        player_rect = pygame.Rect(self.player_pos.x - 10, self.player_pos.y - 20, 20, 30)
        pygame.gfxdraw.filled_circle(self.screen, int(self.player_pos.x), int(self.player_pos.y - 5), 18, self.COLOR_PLAYER_GLOW)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect)
        
        # Darts
        for dart in self.darts:
            pygame.gfxdraw.filled_circle(self.screen, int(dart['pos'].x), int(dart['pos'].y), 3, self.COLOR_DART)
            pygame.gfxdraw.aacircle(self.screen, int(dart['pos'].x), int(dart['pos'].y), 3, self.COLOR_DART)
        
        # Particles
        for p in self.particles:
            size = int(p['life'] / p['max_life'] * p['size'])
            if size > 0:
                pygame.draw.rect(self.screen, p['color'], (p['pos'].x - size//2, p['pos'].y - size//2, size, size))

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        # Darts
        dart_text = self.font_small.render("DARTS:", True, self.COLOR_TEXT)
        self.screen.blit(dart_text, (self.SCREEN_WIDTH - 150, 10))
        for i in range(self.remaining_darts):
            pygame.gfxdraw.filled_trigon(self.screen, 
                self.SCREEN_WIDTH - 80 + i*10, 12,
                self.SCREEN_WIDTH - 80 + i*10, 22,
                self.SCREEN_WIDTH - 75 + i*10, 17,
                self.COLOR_DART)

        # Status
        if self.game_over:
            status_text = ""
            status_color = (200, 200, 50)
            player_rect = pygame.Rect(self.player_pos.x - 10, self.player_pos.y - 20, 20, 30)
            if player_rect.colliderect(pygame.Rect(self.objective_pos.x - 10, self.objective_pos.y - 10, 20, 20)):
                 status_text = "MISSION COMPLETE!"
                 status_color = self.COLOR_OBJECTIVE
            else:
                for guard in self.guards:
                    if guard['state'] == 'alerted':
                        status_text = "DETECTED!"
                        status_color = self.COLOR_GUARD
                        break
                else:
                    status_text = "MISSION FAILED"
            
            if status_text:
                overlay = self.font_large.render(status_text, True, status_color)
                text_rect = overlay.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
                self.screen.blit(overlay, text_rect)
    
    def _create_particles(self, pos, count, color, life=20, size=4):
        for _ in range(count):
            self.particles.append({
                'pos': pos.copy(),
                'vel': pygame.Vector2(random.uniform(-3, 3), random.uniform(-3, 3)),
                'life': life,
                'max_life': life,
                'color': color,
                'size': size
            })

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually
    # It will not run in a headless environment
    try:
        os.environ["SDL_VIDEODRIVER"] = "x11"
        env = GameEnv(render_mode="rgb_array")
        obs, info = env.reset()
        
        screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
        pygame.display.set_caption("Spy Platformer")
        clock = pygame.time.Clock()
        
        done = False
        total_reward = 0
        
        while not done:
            # --- Human Controls ---
            movement = 0 # none
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]: movement = 1
            # elif keys[pygame.K_DOWN]: movement = 2 # Not used
            if keys[pygame.K_LEFT]: movement = 3
            if keys[pygame.K_RIGHT]: movement = 4
            
            space_held = keys[pygame.K_SPACE]
            shift_held = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]
            
            action = [movement, 1 if space_held else 0, 1 if shift_held else 0]
            
            # --- Step Environment ---
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            # --- Pygame Event Handling ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
            
            # --- Render to Screen ---
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            clock.tick(30) # Run at 30 FPS
            
            if terminated or truncated:
                print(f"Episode finished. Total Reward: {total_reward}")
                print("Press 'R' to reset.")

    except pygame.error as e:
        print(f"Could not initialize display for manual play: {e}")
        print("This is normal in a headless environment.")
    finally:
        env.close()