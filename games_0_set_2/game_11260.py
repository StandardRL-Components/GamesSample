import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:42:35.769273
# Source Brief: brief_01260.md
# Brief Index: 1260
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Gymnasium environment for 'Atomic Transporter'.
    The player navigates a decaying atom, using gravity flips and portals
    to collect all subatomic fragments before a timer runs out.
    """
    game_description = (
        "Navigate a decaying atom, using gravity flips and portals to collect all subatomic fragments "
        "before a timer runs out."
    )
    user_guide = (
        "Controls: Use arrow keys to move. Press space to flip gravity. "
        "Press shift to place or move a portal."
    )
    auto_advance = True
    metadata = {"render_modes": ["rgb_array"]}

    # Constants
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    ATOM_RADIUS = 180
    ATOM_CENTER = (WIDTH // 2, HEIGHT // 2)

    # Colors (Vibrant, high contrast)
    COLOR_BG = (20, 10, 40) # Dark Purple
    COLOR_BOUNDARY = (100, 80, 150)
    COLOR_PLAYER = (0, 150, 255)
    COLOR_PLAYER_GLOW = (0, 150, 255, 50)
    COLOR_FRAGMENT = (50, 255, 150)
    COLOR_FRAGMENT_GLOW = (50, 255, 150, 60)
    COLOR_PORTAL_A = (255, 150, 0)
    COLOR_PORTAL_B = (200, 0, 255)
    COLOR_OBSTACLE = (255, 50, 50)
    COLOR_TEXT = (230, 230, 230)
    COLOR_GRAVITY_INDICATOR = (255, 255, 0, 150)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        self.render_mode = render_mode

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 24)

        # Game state variables are initialized in reset()
        self.level = 1
        self.player = None
        self.fragments = []
        self.portals = []
        self.obstacles = []
        self.gravity = None
        self.timer = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.prev_space_held = False
        self.prev_shift_held = False
        self.next_portal_index = 0
        
        # Initial reset is called by the test harness, no need to call it here.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Player entity class defined internally for encapsulation
        class Orb:
            def __init__(self, pos, radius, color, glow_color):
                self.pos = pygame.math.Vector2(pos)
                self.vel = pygame.math.Vector2(0, 0)
                self.radius = radius
                self.color = color
                self.glow_color = glow_color
                self.on_cooldown = 0 # Teleport cooldown

            def apply_force(self, force):
                self.vel += force

            def update(self, gravity, obstacles, boundary_center, boundary_radius):
                if self.on_cooldown > 0:
                    self.on_cooldown -= 1
                
                self.apply_force(gravity)
                self.vel *= 0.98 # Friction/drag
                self.pos += self.vel
                
                # Boundary collision
                to_center = self.pos - boundary_center
                dist_from_center = to_center.length()
                if dist_from_center > boundary_radius - self.radius:
                    self.pos = boundary_center + to_center.normalize() * (boundary_radius - self.radius)
                    self.vel *= -0.5 # Bounce

                # Obstacle collision
                player_rect = pygame.Rect(self.pos.x - self.radius, self.pos.y - self.radius, self.radius * 2, self.radius * 2)
                for obstacle in obstacles:
                    if player_rect.colliderect(obstacle):
                        closest_x = max(obstacle.left, min(self.pos.x, obstacle.right))
                        closest_y = max(obstacle.top, min(self.pos.y, obstacle.bottom))
                        closest_point = pygame.math.Vector2(closest_x, closest_y)
                        
                        push_vec = self.pos - closest_point
                        if push_vec.length_squared() < self.radius**2:
                            dist = push_vec.length()
                            if dist > 0:
                                overlap = self.radius - dist
                                self.pos += push_vec.normalize() * overlap
                                self.vel *= -0.5 # Bounce
                        break

            def draw(self, surface):
                glow_radius = int(self.radius * 1.8)
                s = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(s, self.glow_color, (glow_radius, glow_radius), glow_radius)
                surface.blit(s, (int(self.pos.x - glow_radius), int(self.pos.y - glow_radius)), special_flags=pygame.BLEND_RGBA_ADD)
                
                pygame.gfxdraw.filled_circle(surface, int(self.pos.x), int(self.pos.y), self.radius, self.color)
                pygame.gfxdraw.aacircle(surface, int(self.pos.x), int(self.pos.y), self.radius, self.color)

        self.player = Orb(pos=self.ATOM_CENTER, radius=12, color=self.COLOR_PLAYER, glow_color=self.COLOR_PLAYER_GLOW)
        
        # Reset game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.gravity = pygame.math.Vector2(0, 0.15)
        self.portals = []
        self.next_portal_index = 0
        self.prev_space_held = False
        self.prev_shift_held = False

        # Level setup based on self.level
        num_fragments = 18 + 2 * self.level
        self.timer = 55.0 + 5.0 * self.level
        
        self.fragments = []
        for _ in range(num_fragments):
            while True:
                angle = self.np_random.uniform(0, 2 * math.pi)
                radius = self.np_random.uniform(50, self.ATOM_RADIUS - 20)
                pos = (self.ATOM_CENTER[0] + math.cos(angle) * radius,
                       self.ATOM_CENTER[1] + math.sin(angle) * radius)
                if pygame.math.Vector2(pos).distance_to(self.ATOM_CENTER) > 50:
                    self.fragments.append(pygame.math.Vector2(pos))
                    break
        
        self.obstacles = []
        num_obstacles = min(5, self.level // 2)
        for _ in range(num_obstacles):
            w = self.np_random.integers(40, 101)
            h = self.np_random.integers(10, 21)
            angle = self.np_random.uniform(0, 2 * math.pi)
            radius = self.np_random.uniform(80, self.ATOM_RADIUS - 80)
            pos = (self.ATOM_CENTER[0] + math.cos(angle) * radius - w/2,
                   self.ATOM_CENTER[1] + math.sin(angle) * radius - h/2)
            self.obstacles.append(pygame.Rect(pos[0], pos[1], w, h))

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            obs, info = self.reset()
            return obs, 0, True, False, info

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        space_pressed = space_held and not self.prev_space_held
        shift_pressed = shift_held and not self.prev_shift_held
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        move_force = pygame.math.Vector2(0, 0)
        move_strength = 0.4
        if movement == 1: move_force.y = -move_strength # Up
        elif movement == 2: move_force.y = move_strength # Down
        elif movement == 3: move_force.x = -move_strength # Left
        elif movement == 4: move_force.x = move_strength # Right
        self.player.apply_force(move_force)

        if space_pressed:
            self.gravity *= -1
            # sfx: gravity_flip.wav

        if shift_pressed:
            if len(self.portals) < 2:
                self.portals.append(pygame.math.Vector2(self.player.pos))
            else:
                self.portals[self.next_portal_index] = pygame.math.Vector2(self.player.pos)
            self.next_portal_index = (self.next_portal_index + 1) % 2
            # sfx: portal_open.wav

        self.steps += 1
        self.timer -= 1.0 / self.FPS
        
        prev_dist_to_nearest = self._get_dist_to_nearest_fragment()
        self.player.update(self.gravity, self.obstacles, pygame.math.Vector2(self.ATOM_CENTER), self.ATOM_RADIUS)
        
        if len(self.portals) == 2 and self.player.on_cooldown == 0:
            portal_radius = 20
            if self.player.pos.distance_to(self.portals[0]) < portal_radius:
                self.player.pos = pygame.math.Vector2(self.portals[1])
                self.player.on_cooldown = 20 # Teleport cooldown
                # sfx: teleport.wav
            elif self.player.pos.distance_to(self.portals[1]) < portal_radius:
                self.player.pos = pygame.math.Vector2(self.portals[0])
                self.player.on_cooldown = 20
                # sfx: teleport.wav

        reward = 0
        collected_fragments = []
        for frag in self.fragments:
            if self.player.pos.distance_to(frag) < self.player.radius + 8:
                collected_fragments.append(frag)
                self.score += 1
                reward += 1.0
                # sfx: collect_fragment.wav
        
        self.fragments = [f for f in self.fragments if f not in collected_fragments]
        
        reward -= 0.01

        new_dist_to_nearest = self._get_dist_to_nearest_fragment()
        if new_dist_to_nearest is not None and prev_dist_to_nearest is not None:
            reward += 0.1 * (prev_dist_to_nearest - new_dist_to_nearest)

        terminated = False
        if not self.fragments: # Victory
            reward += 100.0
            terminated = True
            self.game_over = True
            self.level += 1
        elif self.timer <= 0: # Failure
            reward -= 100.0
            terminated = True
            self.game_over = True
            self.level = 1
        elif self.steps >= 2700: # Max steps (90s @ 30fps)
            terminated = True
            self.game_over = True
            self.level = 1

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _get_dist_to_nearest_fragment(self):
        if not self.fragments:
            return None
        return min(self.player.pos.distance_to(f) for f in self.fragments)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        pygame.gfxdraw.aacircle(self.screen, self.ATOM_CENTER[0], self.ATOM_CENTER[1], self.ATOM_RADIUS, self.COLOR_BOUNDARY)
        pygame.gfxdraw.aacircle(self.screen, self.ATOM_CENTER[0], self.ATOM_CENTER[1], self.ATOM_RADIUS-1, self.COLOR_BOUNDARY)
        
        for obstacle in self.obstacles:
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, obstacle)

        if len(self.portals) > 0: self._draw_portal(self.portals[0], self.COLOR_PORTAL_A)
        if len(self.portals) > 1: self._draw_portal(self.portals[1], self.COLOR_PORTAL_B)

        for frag in self.fragments:
            pos_int = (int(frag.x), int(frag.y))
            s = pygame.Surface((32, 32), pygame.SRCALPHA)
            pygame.draw.circle(s, self.COLOR_FRAGMENT_GLOW, (16, 16), 16)
            self.screen.blit(s, (pos_int[0] - 16, pos_int[1] - 16), special_flags=pygame.BLEND_RGBA_ADD)
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], 8, self.COLOR_FRAGMENT)
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], 8, self.COLOR_FRAGMENT)

        self.player.draw(self.screen)
    
    def _draw_portal(self, pos, color):
        p_int = (int(pos.x), int(pos.y))
        base_radius = 20
        for i in range(4):
            radius = base_radius + 5 * math.sin(self.steps * 0.05 + i)
            alpha = 100 + 150 * abs(math.sin(self.steps * 0.05 + i * 0.5))
            start_angle = (self.steps * (i + 1) * 0.1) % (2 * math.pi)
            end_angle = start_angle + math.pi * 1.5
            
            s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            try:
                pygame.draw.arc(s, (*color, int(alpha)), (p_int[0] - radius, p_int[1] - radius, radius*2, radius*2), start_angle, end_angle, 3)
                self.screen.blit(s, (0,0))
            except (ValueError, TypeError): # Catch potential errors with invalid rects
                pass

    def _render_ui(self):
        score_text = self.font_large.render(f"{self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 10))
        frag_icon = pygame.Surface((16, 16), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(frag_icon, 8, 8, 8, self.COLOR_FRAGMENT)
        pygame.gfxdraw.aacircle(frag_icon, 8, 8, 8, self.COLOR_FRAGMENT)
        self.screen.blit(frag_icon, (score_text.get_width() + 30, 22))

        timer_str = f"{max(0, self.timer):.1f}"
        timer_color = self.COLOR_TEXT if self.timer > 10 else self.COLOR_OBSTACLE
        timer_text = self.font_large.render(timer_str, True, timer_color)
        self.screen.blit(timer_text, (self.WIDTH - timer_text.get_width() - 20, 10))

        indicator_pos = pygame.math.Vector2(self.WIDTH - 40, self.HEIGHT - 40)
        grav_dir = self.gravity.normalize()
        p1 = indicator_pos - grav_dir * 15
        p2 = indicator_pos + grav_dir * 15
        p3 = indicator_pos + grav_dir.rotate(90) * 8
        p4 = indicator_pos + grav_dir.rotate(-90) * 8
        points = [(p2.x, p2.y), (p3.x, p3.y), (p4.x, p4.y)]
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_GRAVITY_INDICATOR)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_GRAVITY_INDICATOR)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "timer": self.timer,
            "level": self.level,
            "fragments_left": len(self.fragments)
        }

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    env = GameEnv()
    
    display_screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Atomic Transporter - Manual Play")
    
    obs, info = env.reset(seed=42)
    running = True
    game_done = False
    
    while running:
        action = [0, 0, 0] # Default action: no-op
        
        keys = pygame.key.get_pressed()
        if not game_done:
            if keys[pygame.K_UP] or keys[pygame.K_w]: action[0] = 1
            elif keys[pygame.K_DOWN] or keys[pygame.K_s]: action[0] = 2
            elif keys[pygame.K_LEFT] or keys[pygame.K_a]: action[0] = 3
            elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: action[0] = 4
            
            action[1] = 1 if keys[pygame.K_SPACE] else 0
            action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            obs, reward, terminated, truncated, info = env.step(action)
            game_done = terminated or truncated

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs, info = env.reset(seed=42)
                    game_done = False
                if event.key == pygame.K_q:
                    running = False

        if game_done:
            end_font = pygame.font.Font(None, 50)
            msg = "VICTORY!" if info['fragments_left'] == 0 else "TIME UP!"
            end_text = end_font.render(msg, True, (255, 255, 255))
            text_rect = end_text.get_rect(center=(env.WIDTH/2, env.HEIGHT/2 - 20))
            display_screen.blit(end_text, text_rect)
            
            sub_font = pygame.font.Font(None, 30)
            sub_text = sub_font.render("Press 'R' to restart or 'Q' to quit", True, (200, 200, 200))
            sub_rect = sub_text.get_rect(center=(env.WIDTH/2, env.HEIGHT/2 + 20))
            display_screen.blit(sub_text, sub_rect)
            pygame.display.flip()
            
        env.clock.tick(env.FPS)
        
    env.close()