import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:03:19.913334
# Source Brief: brief_00783.md
# Brief Index: 783
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class Particle:
    def __init__(self, x, y, vx, vy, color, size, lifespan):
        self.pos = pygame.Vector2(x, y)
        self.vel = pygame.Vector2(vx, vy)
        self.color = color
        self.size = size
        self.lifespan = lifespan
        self.max_lifespan = lifespan

    def update(self):
        self.pos += self.vel
        self.lifespan -= 1
        self.size = max(0, self.size - 0.1)

    def draw(self, surface):
        if self.lifespan > 0:
            alpha = int(255 * (self.lifespan / self.max_lifespan))
            color = self.color + (alpha,)
            temp_surf = pygame.Surface((self.size * 2, self.size * 2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (self.size, self.size), self.size)
            surface.blit(temp_surf, self.pos - pygame.Vector2(self.size, self.size))

class PatrolBot:
    def __init__(self, path_start, path_end, speed):
        self.pos = pygame.Vector2(path_start)
        self.start = pygame.Vector2(path_start)
        self.end = pygame.Vector2(path_end)
        self.speed = speed
        self.direction = 1
        self.rect = pygame.Rect(self.pos.x - 10, self.pos.y - 15, 20, 30)

    def update(self):
        target = self.end if self.direction == 1 else self.start
        move_vec = target - self.pos
        if move_vec.length() < self.speed:
            self.pos = target
            self.direction *= -1
        else:
            self.pos += move_vec.normalize() * self.speed
        self.rect.center = self.pos

    def draw(self, surface):
        # Body
        pygame.draw.rect(surface, (100, 110, 120), self.rect, border_radius=3)
        # Glowing eye
        eye_pos = (int(self.rect.centerx), int(self.rect.centery - 5))
        GameEnv.draw_glowing_circle(surface, GameEnv.COLOR_DANGER, eye_pos, 4, 10)

class SecurityCamera:
    def __init__(self, pos, base_angle, scan_arc, scan_speed, view_dist):
        self.pos = pygame.Vector2(pos)
        self.base_angle = math.radians(base_angle)
        self.scan_arc = math.radians(scan_arc)
        self.scan_speed = scan_speed
        self.view_dist = view_dist
        self.current_angle = self.base_angle
        self.time = 0
        self.active = True
        self.disabled_timer = 0
        self.rect = pygame.Rect(self.pos.x - 8, self.pos.y - 8, 16, 16)

    def update(self, time_step):
        if self.active:
            self.time += time_step * self.scan_speed
            self.current_angle = self.base_angle + math.sin(self.time) * (self.scan_arc / 2)
        else:
            self.disabled_timer -= 1
            if self.disabled_timer <= 0:
                self.active = True

    def disable(self, duration=300):
        self.active = False
        self.disabled_timer = duration

    def check_detection(self, player_rect):
        if not self.active:
            return False
        
        player_pos = pygame.Vector2(player_rect.center)
        vec_to_player = player_pos - self.pos
        dist_to_player = vec_to_player.length()

        if dist_to_player > self.view_dist or dist_to_player == 0:
            return False

        angle_to_player = math.atan2(vec_to_player.y, vec_to_player.x)
        angle_diff = (self.current_angle - angle_to_player + math.pi) % (2 * math.pi) - math.pi

        return abs(angle_diff) < math.radians(20) # 40 degree cone width

    def draw(self, surface):
        # Body
        body_color = (120, 130, 140) if self.active else (60, 60, 60)
        pygame.draw.rect(surface, body_color, self.rect)
        pygame.draw.circle(surface, body_color, (int(self.pos.x), int(self.pos.y)), 10)

        # Lens
        lens_color = GameEnv.COLOR_DANGER if self.active else (40, 0, 0)
        GameEnv.draw_glowing_circle(surface, lens_color, (int(self.pos.x), int(self.pos.y)), 3, 6)

        # Vision Cone
        if self.active:
            angle1 = self.current_angle - math.radians(20)
            angle2 = self.current_angle + math.radians(20)
            p1 = self.pos
            p2 = self.pos + self.view_dist * pygame.Vector2(math.cos(angle1), math.sin(angle1))
            p3 = self.pos + self.view_dist * pygame.Vector2(math.cos(angle2), math.sin(angle2))
            
            # Flicker effect
            alpha = random.randint(30, 50)
            pygame.gfxdraw.aapolygon(surface, [p1, p2, p3], GameEnv.COLOR_DANGER + (alpha,))
            pygame.gfxdraw.filled_polygon(surface, [p1, p2, p3], GameEnv.COLOR_DANGER + (alpha,))

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Infiltrate a high-tech facility as a shapeshifting operative. Evade security in human form or transform into an automaton to pass safely."
    )
    user_guide = (
        "Controls: Use ←→ to move and ↑ to jump. Press space to switch forms. Press shift near terminals to disable cameras."
    )
    auto_advance = True

    # --- Colors ---
    COLOR_BG = (15, 20, 30)
    COLOR_PLATFORM = (40, 50, 70)
    COLOR_PLATFORM_NEON = (70, 180, 220)
    COLOR_PLAYER_HUMAN = (200, 255, 255)
    COLOR_PLAYER_AUTOMATON = (255, 200, 0)
    COLOR_DANGER = (255, 50, 50)
    COLOR_INTERACT = (50, 150, 255)
    COLOR_EXIT = (50, 255, 150)
    COLOR_UI_TEXT = (220, 220, 220)

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    GRAVITY = 0.4
    PLAYER_SPEED = 4.0
    PLAYER_JUMP_HUMAN = -7.0
    PLAYER_JUMP_AUTOMATON = -11.0
    MAX_STEPS = 1500
    TOTAL_ZONES = 5

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Consolas', 20, bold=True)
        
        # Game state variables initialized in reset
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.current_zone = 1
        self.player_pos = pygame.Vector2(0, 0)
        self.player_vel = pygame.Vector2(0, 0)
        self.player_form = 'human'
        self.on_ground = False
        self.player_rect = pygame.Rect(0, 0, 20, 40)
        self.prev_space_held = False
        self.prev_shift_held = False
        self.platforms = []
        self.bots = []
        self.cameras = []
        self.short_circuits = []
        self.scraps = []
        self.exit_rect = pygame.Rect(0, 0, 0, 0)
        self.particles = []
        self.scrap_collected = 0
        self.background_stars = []

        self.reset()
    
    def _generate_zone(self):
        self.platforms = [pygame.Rect(0, self.HEIGHT - 20, self.WIDTH, 20)]
        self.bots = []
        self.cameras = []
        self.short_circuits = []
        self.scraps = []
        self.exit_rect = pygame.Rect(self.WIDTH - 60, 30, 40, 40)
        
        # Procedural platform generation
        current_pos = pygame.Vector2(80, self.HEIGHT - 50)
        for _ in range(15):
            width = random.randint(80, 150)
            height = 20
            self.platforms.append(pygame.Rect(current_pos.x, current_pos.y, width, height))
            current_pos.x += width + random.randint(40, 100)
            current_pos.y -= random.randint(-40, 70)
            if current_pos.x > self.WIDTH - 150:
                current_pos.x = random.randint(50, 150)
                current_pos.y -= random.randint(60, 100)
            current_pos.y = max(50, current_pos.y)

        # Place entities based on zone
        num_bots = min(4, self.current_zone)
        num_cameras = min(5, self.current_zone)
        
        valid_platforms = [p for p in self.platforms if p.y < self.HEIGHT - 30 and p.width > 100]
        random.shuffle(valid_platforms)

        for i in range(num_bots):
            if not valid_platforms: break
            p = valid_platforms.pop()
            speed = 1.0 + self.current_zone * 0.1
            self.bots.append(PatrolBot((p.left, p.top - 20), (p.right, p.top - 20), speed))

        for i in range(num_cameras):
            if not valid_platforms: break
            p = valid_platforms.pop()
            pos = (random.randint(p.left, p.right), p.top - 10)
            self.cameras.append(SecurityCamera(pos, random.randint(180, 360), 90, 0.5, 200))
            # Add a short-circuit point nearby
            if random.random() < 0.7:
                sc_pos = (pos[0] + random.randint(-50, 50), pos[1] + random.randint(10, 30))
                self.short_circuits.append({'pos': pygame.Vector2(sc_pos), 'camera': self.cameras[-1]})

        for _ in range(5): # Add scrap
            if not valid_platforms: break
            p = valid_platforms.pop()
            self.scraps.append(pygame.Rect(p.centerx - 5, p.top - 15, 10, 10))
            
        # Background elements
        self.background_stars = [(random.randint(0, self.WIDTH), random.randint(0, self.HEIGHT), random.randint(1, 2)) for _ in range(100)]


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.current_zone = 1
        self.scrap_collected = 0
        
        self._generate_zone()
        
        self.player_pos = pygame.Vector2(80, self.HEIGHT - 80)
        self.player_vel = pygame.Vector2(0, 0)
        self.player_form = 'human'
        self.on_ground = False
        self.prev_space_held = True # Prevent action on first frame
        self.prev_shift_held = True

        self.particles.clear()
        
        return self._get_observation(), self._get_info()
    
    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- Horizontal Movement ---
        if movement == 3: # Left
            self.player_vel.x = -self.PLAYER_SPEED
        elif movement == 4: # Right
            self.player_vel.x = self.PLAYER_SPEED
        else:
            self.player_vel.x = 0

        # --- Vertical Movement (Jump) ---
        if movement == 1 and self.on_ground: # Up
            jump_strength = self.PLAYER_JUMP_AUTOMATON if self.player_form == 'automaton' else self.PLAYER_JUMP_HUMAN
            self.player_vel.y = jump_strength
            self.on_ground = False
            # sound: player_jump.wav
            self._spawn_particles(self.player_rect.midbottom, 15, (200, 200, 255), 2.0, 3.0)

        # --- Form Switch (Space Press) ---
        reward_bonus = 0
        if space_held and not self.prev_space_held:
            self.player_form = 'automaton' if self.player_form == 'human' else 'human'
            # sound: form_switch.wav
            color = self.COLOR_PLAYER_AUTOMATON if self.player_form == 'automaton' else self.COLOR_PLAYER_HUMAN
            self._spawn_particles(self.player_rect.center, 30, color, 3.0, 5.0)

        # --- Short Circuit (Shift Press) ---
        if shift_held and not self.prev_shift_held:
            for sc in self.short_circuits:
                if self.player_pos.distance_to(sc['pos']) < 40 and sc['camera'].active:
                    sc['camera'].disable()
                    reward_bonus += 1.0
                    # sound: short_circuit.wav
                    self._spawn_particles(sc['pos'], 20, self.COLOR_INTERACT, 2.0, 4.0)
                    self._spawn_particles(sc['camera'].pos, 10, self.COLOR_INTERACT, 1.0, 2.0)
                    break # Only one per press

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held
        return reward_bonus

    def _update_game_state(self):
        # --- Player Physics ---
        self.player_vel.y += self.GRAVITY
        self.player_pos.x += self.player_vel.x
        self.player_rect.x = int(self.player_pos.x)
        
        # Horizontal collision
        for plat in self.platforms:
            if self.player_rect.colliderect(plat):
                if self.player_vel.x > 0: self.player_rect.right = plat.left
                elif self.player_vel.x < 0: self.player_rect.left = plat.right
                self.player_pos.x = self.player_rect.x

        self.player_pos.y += self.player_vel.y
        self.player_rect.y = int(self.player_pos.y)

        # Vertical collision
        self.on_ground = False
        for plat in self.platforms:
            if self.player_rect.colliderect(plat):
                if self.player_vel.y > 0:
                    self.player_rect.bottom = plat.top
                    self.player_vel.y = 0
                    if not self.on_ground: # Landing
                        # sound: player_land.wav
                        self._spawn_particles(self.player_rect.midbottom, 5, (200,200,200), 1.0, 2.0)
                    self.on_ground = True
                elif self.player_vel.y < 0:
                    self.player_rect.top = plat.bottom
                    self.player_vel.y = 0
                self.player_pos.y = self.player_rect.y

        # Keep player in bounds
        self.player_pos.x = np.clip(self.player_pos.x, 0, self.WIDTH - self.player_rect.width)
        self.player_rect.x = int(self.player_pos.x)
        if self.player_pos.y > self.HEIGHT: # Fell off screen
            self.game_over = True

        # --- Update Entities ---
        time_step = self.clock.get_time() / 1000.0
        for bot in self.bots: bot.update()
        for cam in self.cameras: cam.update(time_step)
        for particle in self.particles: particle.update()
        self.particles = [p for p in self.particles if p.lifespan > 0]

    def _check_events_and_rewards(self):
        reward = 0.1 # Survival reward

        # --- Scrap Collection ---
        collected_indices = self.player_rect.collidelistall(self.scraps)
        if collected_indices:
            for i in sorted(collected_indices, reverse=True):
                # sound: scrap_collect.wav
                self._spawn_particles(self.scraps[i].center, 10, self.COLOR_INTERACT, 1.0, 2.0)
                del self.scraps[i]
                self.scrap_collected += 1
                reward += 0.5

        # --- Exit Zone ---
        if self.player_rect.colliderect(self.exit_rect):
            # sound: zone_complete.wav
            if self.current_zone >= self.TOTAL_ZONES:
                self.score += 100
                reward += 100
                self.game_over = True
            else:
                self.current_zone += 1
                self.score += 5
                reward += 5
                self._generate_zone()
                self.player_pos = pygame.Vector2(80, self.HEIGHT - 80)
                self.player_vel = pygame.Vector2(0, 0)
                self._spawn_particles(self.player_rect.center, 50, self.COLOR_EXIT, 5.0, 8.0)

        # --- Detection ---
        if self.player_form == 'human' and not self.game_over:
            detected = False
            # Bot collision
            for bot in self.bots:
                if self.player_rect.colliderect(bot.rect):
                    detected = True
                    break
            # Camera detection
            if not detected:
                for cam in self.cameras:
                    if cam.check_detection(self.player_rect):
                        detected = True
                        break
            
            if detected:
                # sound: detection_alarm.wav
                self.score -= 100
                reward = -100
                self.game_over = True
                self._spawn_particles(self.player_rect.center, 50, self.COLOR_DANGER, 4.0, 6.0)
        
        return reward

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
        
        reward = self._handle_input(action)
        self._update_game_state()
        reward += self._check_events_and_rewards()
        
        self.score += reward
        self.steps += 1
        
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        if terminated and not self.game_over: # Timeout
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _render_game(self):
        # Background
        for x, y, size in self.background_stars:
            pygame.draw.rect(self.screen, (200, 200, 255, 50), (x, y, size, size))
        
        # Platforms
        for plat in self.platforms:
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, plat)
            pygame.draw.line(self.screen, self.COLOR_PLATFORM_NEON, plat.topleft, plat.topright, 2)

        # Exit
        self.draw_glowing_circle(self.screen, self.COLOR_EXIT, self.exit_rect.center, 15, 30)
        
        # Interactables
        for sc in self.short_circuits:
            if sc['camera'].active:
                self.draw_glowing_circle(self.screen, self.COLOR_INTERACT, sc['pos'], 5, 12)
        for scrap in self.scraps:
            self.draw_glowing_circle(self.screen, self.COLOR_INTERACT, scrap.center, 4, 10)

        # Entities
        for bot in self.bots: bot.draw(self.screen)
        for cam in self.cameras: cam.draw(self.screen)
        
        # Player
        if self.player_form == 'human':
            self.player_rect.height = 40
            self.player_rect.width = 20
            color = self.COLOR_PLAYER_HUMAN
            self.draw_glowing_circle(self.screen, color, self.player_rect.center, 15, 30)
            pygame.draw.rect(self.screen, color, self.player_rect, border_radius=5)
        else: # Automaton
            self.player_rect.height = 30
            self.player_rect.width = 30
            color = self.COLOR_PLAYER_AUTOMATON
            self.draw_glowing_circle(self.screen, color, self.player_rect.center, 20, 40)
            pygame.draw.rect(self.screen, color, self.player_rect, border_radius=8)

        # Particles
        for particle in self.particles: particle.draw(self.screen)

    def _render_ui(self):
        score_text = self.font.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        zone_text = self.font.render(f"ZONE: {self.current_zone}/{self.TOTAL_ZONES}", True, self.COLOR_UI_TEXT)
        scrap_text = self.font.render(f"SCRAP: {self.scrap_collected}", True, self.COLOR_UI_TEXT)
        
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(zone_text, (self.WIDTH - zone_text.get_width() - 10, 10))
        self.screen.blit(scrap_text, (10, 35))

        # Form indicator
        form_icon_pos = (self.WIDTH - 40, self.HEIGHT - 40)
        if self.player_form == 'human':
            pygame.draw.rect(self.screen, self.COLOR_PLAYER_HUMAN, (*form_icon_pos, 20, 20), 2, border_radius=3)
        else:
            pygame.draw.rect(self.screen, self.COLOR_PLAYER_AUTOMATON, (*form_icon_pos, 20, 20), border_radius=5)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "zone": self.current_zone,
            "player_form": self.player_form,
            "scrap": self.scrap_collected,
        }

    def _spawn_particles(self, pos, count, color, min_speed, max_speed):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(min_speed, max_speed)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            size = random.uniform(2, 5)
            lifespan = random.randint(15, 30)
            self.particles.append(Particle(pos[0], pos[1], vx, vy, color, size, lifespan))

    @staticmethod
    def draw_glowing_circle(surface, color, center, radius, glow_radius):
        # Draw glow
        glow_color = color + (20,)
        temp_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(temp_surf, glow_color, (glow_radius, glow_radius), glow_radius)
        surface.blit(temp_surf, (center[0] - glow_radius, center[1] - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)
        # Draw main circle
        pygame.draw.circle(surface, color, center, radius)

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # --- Manual Control Mapping ---
    # Arrows: Move
    # Space: Switch Form
    # Left Shift: Short Circuit
    
    total_reward = 0
    # Create a display surface if it doesn't exist
    pygame.display.init()
    display_surf = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    
    while not done:
        # Default action
        movement, space, shift = 0, 0, 0
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        if keys[pygame.K_DOWN]: movement = 2
        if keys[pygame.K_LEFT]: movement = 3
        if keys[pygame.K_RIGHT]: movement = 4
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
        
        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        
        # Render for human viewing
        img = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(img)
        
        display_surf.blit(surf, (0, 0))
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        env.clock.tick(30)

    print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
    pygame.quit()