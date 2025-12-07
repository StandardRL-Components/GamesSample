import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:55:01.970439
# Source Brief: brief_02036.md
# Brief Index: 2036
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math

class GameEnv(gym.Env):
    """
    Navigate the treacherous interior of a neutron star, shrinking to stealthily 
    avoid dangers and growing to combat threats while terraforming magnetic 
    fields to reach the core.
    """
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Navigate the treacherous interior of a neutron star. Shrink to avoid dangers, "
        "grow your magnetic field to repel them, and find portals to journey deeper to the core."
    )
    user_guide = (
        "Use arrow keys to move. Hold Space to shrink (faster, smaller). "
        "Hold Shift to grow (slower, larger magnetic field)."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        
        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16)
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)

        # Colors
        self.COLOR_BG = (10, 5, 25)
        self.COLOR_PLAYER = (0, 255, 150)
        self.COLOR_HAZARD = (255, 50, 50)
        self.COLOR_PORTAL = (180, 0, 255)
        self.COLOR_MAGNETIC = (255, 220, 0)
        self.COLOR_UI_TEXT = (220, 220, 255)
        self.COLOR_UNSTABLE = (100, 20, 20)
        
        # Game constants
        self.MAX_STEPS = 5000
        self.INITIAL_DISTANCE_TO_CORE = 2000
        self.PLAYER_CENTER = pygame.math.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        self.PLAYER_MIN_SIZE = 10
        self.PLAYER_MAX_SIZE = 40
        self.PLAYER_SIZE_CHANGE_RATE = 0.5
        
        # State variables are initialized in reset()
        self.world_pos = None
        self.player_size = None
        self.magnetic_influence_radius = None
        self.distance_to_core = None
        self.hazard_frequency = None
        self.is_unstable_region = None
        self.unstable_region_timer = None
        
        self.hazards = []
        self.particles = []
        self.portal_pos = None
        self.bg_stars = []

        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # Initialize state with a dummy seed for the first time
        self.reset(seed=0)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.world_pos = pygame.math.Vector2(0, 0)
        self.player_size = (self.PLAYER_MIN_SIZE + self.PLAYER_MAX_SIZE) / 2
        self.magnetic_influence_radius = 80
        self.distance_to_core = self.INITIAL_DISTANCE_TO_CORE
        self.hazard_frequency = 0.02
        self.unstable_region_timer = 0
        
        self.hazards.clear()
        self.particles.clear()
        
        self._generate_new_region()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0

        # --- Handle Actions & Update Player ---
        reward += self._handle_actions(action)
        
        # --- Update Game World ---
        self._update_world_state()
        
        # --- Calculate Rewards ---
        reward += self._calculate_rewards()

        self.score += reward
        
        # --- Check Termination ---
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS

        if terminated:
            self.game_over = True
            if self.distance_to_core <= 0:
                reward += 100 # Victory
            else:
                reward -= 100 # Defeat
                self._spawn_particle(self.PLAYER_CENTER, 100, self.COLOR_HAZARD, 50, 8)
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_actions(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        action_reward = 0

        # Size change
        size_changed = False
        if space_held and not shift_held: # Shrink
            new_size = self.player_size - self.PLAYER_SIZE_CHANGE_RATE
            self.player_size = max(self.PLAYER_MIN_SIZE, new_size)
            size_changed = True
        elif shift_held and not space_held: # Grow
            new_size = self.player_size + self.PLAYER_SIZE_CHANGE_RATE
            self.player_size = min(self.PLAYER_MAX_SIZE, new_size)
            size_changed = True
        
        if size_changed:
            action_reward -= 1.0 # Penalty for changing size
            self._spawn_particle(self.PLAYER_CENTER, 5, self.COLOR_PLAYER, 10, 1.5, 0.5)

        # Movement
        # Speed is inversely proportional to size
        speed = 2 + (self.PLAYER_MAX_SIZE - self.player_size) * 0.2
        move_vector = pygame.math.Vector2(0, 0)
        if movement == 1: move_vector.y = -1 # Up
        elif movement == 2: move_vector.y = 1 # Down
        elif movement == 3: move_vector.x = -1 # Left
        elif movement == 4: move_vector.x = 1 # Right
        
        if move_vector.length() > 0:
            self.world_pos += move_vector * speed
            self.distance_to_core -= move_vector.length() * speed * 0.1
        
        return action_reward

    def _update_world_state(self):
        self.hazard_frequency += 0.0001
        self._update_hazards()
        self._update_particles()
        
        if self.is_unstable_region:
            self.unstable_region_timer += 1
            if self.np_random.random() < 0.2:
                self._spawn_particle(
                    (self.np_random.uniform(0, self.WIDTH), self.np_random.uniform(0, self.HEIGHT)),
                    1, self.COLOR_UNSTABLE, 20, 1, 0
                )
        else:
            self.unstable_region_timer = 0

    def _update_hazards(self):
        # Spawn new hazards
        if self.np_random.random() < self.hazard_frequency:
            edge = self.np_random.integers(4)
            if edge == 0: # Top
                pos = pygame.math.Vector2(self.np_random.uniform(-50, self.WIDTH + 50), -50)
            elif edge == 1: # Bottom
                pos = pygame.math.Vector2(self.np_random.uniform(-50, self.WIDTH + 50), self.HEIGHT + 50)
            elif edge == 2: # Left
                pos = pygame.math.Vector2(-50, self.np_random.uniform(-50, self.HEIGHT + 50))
            else: # Right
                pos = pygame.math.Vector2(self.WIDTH + 50, self.np_random.uniform(-50, self.HEIGHT + 50))
            
            # Convert to world coordinates
            world_pos = pos + self.world_pos
            self.hazards.append({'pos': world_pos, 'radius': self.np_random.uniform(8, 15)})

        # Update existing hazards
        for hazard in self.hazards[:]:
            screen_pos = hazard['pos'] - self.world_pos
            
            # Magnetic repulsion
            to_player = self.PLAYER_CENTER - screen_pos
            dist_to_player = to_player.length()
            
            if dist_to_player < self.magnetic_influence_radius:
                repel_strength = (1 - (dist_to_player / self.magnetic_influence_radius))
                repel_force = to_player.normalize() * repel_strength * (self.player_size / self.PLAYER_MIN_SIZE) * 0.5
                hazard['pos'] += repel_force
                if self.np_random.random() < 0.1:
                    self._spawn_particle(screen_pos, 3, self.COLOR_MAGNETIC, 15, 2)

            # Movement towards player
            if dist_to_player > 0:
                direction = (self.PLAYER_CENTER - screen_pos).normalize()
                hazard['pos'] += direction * 1.5 # Move in world space

            # Remove if too far
            if dist_to_player > max(self.WIDTH, self.HEIGHT) * 1.5:
                self.hazards.remove(hazard)

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _calculate_rewards(self):
        reward = 0.1 # Survival reward

        # Unstable region penalty
        if self.is_unstable_region:
            reward -= 0.5

        # Check for portal collision
        portal_screen_pos = self.portal_pos - self.world_pos
        if portal_screen_pos.distance_to(self.PLAYER_CENTER) < 30 + self.player_size:
            reward += 5.0
            self.magnetic_influence_radius += 5
            self.distance_to_core -= 250
            self._generate_new_region()
            self._spawn_particle(self.PLAYER_CENTER, 50, self.COLOR_PORTAL, 40, 10)
        
        # Check for hazard collision
        for hazard in self.hazards:
            hazard_screen_pos = hazard['pos'] - self.world_pos
            if hazard_screen_pos.distance_to(self.PLAYER_CENTER) < hazard['radius'] + self.player_size:
                self.game_over = True
                break
        
        return reward

    def _check_termination(self):
        if self.game_over: return True
        if self.distance_to_core <= 0: return True
        if self.unstable_region_timer >= 100: return True
        return False
        
    def _get_observation(self):
        self._render_all()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "distance_to_core": self.distance_to_core,
            "player_size": self.player_size,
            "magnetic_influence": self.magnetic_influence_radius
        }

    def _generate_new_region(self):
        self.portal_pos = self.world_pos + pygame.math.Vector2(
            self.np_random.uniform(-self.WIDTH * 1.5, self.WIDTH * 1.5),
            self.np_random.uniform(-self.HEIGHT * 1.5, self.HEIGHT * 1.5)
        )
        # Ensure portal is not too close
        while self.portal_pos.distance_to(self.world_pos) < self.WIDTH / 2:
             self.portal_pos = self.world_pos + pygame.math.Vector2(
                self.np_random.uniform(-self.WIDTH * 1.5, self.WIDTH * 1.5),
                self.np_random.uniform(-self.HEIGHT * 1.5, self.HEIGHT * 1.5)
            )

        self.is_unstable_region = self.np_random.random() < 0.25
        self.unstable_region_timer = 0
        self.hazards.clear()
        
        # Generate static background stars for this region
        self.bg_stars.clear()
        for _ in range(200):
            self.bg_stars.append({
                'pos': pygame.math.Vector2(self.np_random.uniform(-self.WIDTH, self.WIDTH*2),
                                           self.np_random.uniform(-self.HEIGHT, self.HEIGHT*2)),
                'depth': self.np_random.uniform(0.1, 0.6),
                'color': (
                    self.np_random.integers(20, 60),
                    self.np_random.integers(10, 40),
                    self.np_random.integers(40, 80)
                )
            })

    def _render_all(self):
        # Background
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        
        # Game elements
        self._render_portal()
        self._render_hazards()
        self._render_particles()
        self._render_player()

        # UI
        self._render_ui()

    def _render_background(self):
        for star in self.bg_stars:
            # Parallax scrolling
            screen_x = (star['pos'].x - self.world_pos.x * star['depth']) % self.WIDTH
            screen_y = (star['pos'].y - self.world_pos.y * star['depth']) % self.HEIGHT
            size = int(star['depth'] * 3)
            pygame.draw.circle(self.screen, star['color'], (int(screen_x), int(screen_y)), size)

    def _render_portal(self):
        pos = self.portal_pos - self.world_pos
        if -50 < pos.x < self.WIDTH + 50 and -50 < pos.y < self.HEIGHT + 50:
            t = self.steps * 0.1
            radius1 = 30 + 5 * math.sin(t)
            radius2 = 30 + 5 * math.cos(t + math.pi / 2)
            
            color1 = (
                max(0, min(255, self.COLOR_PORTAL[0] + 20 * math.sin(t))),
                self.COLOR_PORTAL[1],
                max(0, min(255, self.COLOR_PORTAL[2] - 20 * math.sin(t)))
            )
            color2 = (
                max(0, min(255, self.COLOR_PORTAL[0] - 20 * math.cos(t))),
                self.COLOR_PORTAL[1],
                max(0, min(255, self.COLOR_PORTAL[2] + 20 * math.cos(t)))
            )
            
            pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), int(radius1), color1)
            pygame.gfxdraw.aacircle(self.screen, int(pos.x), int(pos.y), int(radius1), color1)
            pygame.gfxdraw.aacircle(self.screen, int(pos.x), int(pos.y), int(radius2), color2)

    def _render_hazards(self):
        for hazard in self.hazards:
            pos = hazard['pos'] - self.world_pos
            if -50 < pos.x < self.WIDTH + 50 and -50 < pos.y < self.HEIGHT + 50:
                pulse = 1 + 0.1 * math.sin(self.steps * 0.2 + pos.x)
                radius = int(hazard['radius'] * pulse)
                color = (
                    self.COLOR_HAZARD[0],
                    int(self.COLOR_HAZARD[1] * pulse),
                    int(self.COLOR_HAZARD[2] * pulse)
                )
                pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), radius, color)
                pygame.gfxdraw.aacircle(self.screen, int(pos.x), int(pos.y), radius, color)

    def _render_player(self):
        mag_color = (*self.COLOR_MAGNETIC, 20 + int(20 * (self.player_size / self.PLAYER_MAX_SIZE)))
        pygame.gfxdraw.filled_circle(self.screen, int(self.PLAYER_CENTER.x), int(self.PLAYER_CENTER.y), int(self.magnetic_influence_radius), mag_color)
        pygame.gfxdraw.aacircle(self.screen, int(self.PLAYER_CENTER.x), int(self.PLAYER_CENTER.y), int(self.magnetic_influence_radius), mag_color)

        for i in range(4):
            glow_alpha = 80 - i * 20
            glow_size = self.player_size + i * 3
            glow_color = (*self.COLOR_PLAYER, glow_alpha)
            pygame.gfxdraw.filled_circle(self.screen, int(self.PLAYER_CENTER.x), int(self.PLAYER_CENTER.y), int(glow_size), glow_color)
        
        pygame.gfxdraw.filled_circle(self.screen, int(self.PLAYER_CENTER.x), int(self.PLAYER_CENTER.y), int(self.player_size), self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, int(self.PLAYER_CENTER.x), int(self.PLAYER_CENTER.y), int(self.player_size), self.COLOR_PLAYER)

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = (*p['color'], alpha)
            try:
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), int(p['radius']), color)
            except (OverflowError, ValueError):
                pass

    def _render_ui(self):
        if self.is_unstable_region:
            alpha = 50 + 40 * math.sin(self.steps * 0.3)
            s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            s.fill((*self.COLOR_UNSTABLE, alpha))
            self.screen.blit(s, (0,0))
            
            warning_text = self.font_large.render("! UNSTABLE REGION !", True, self.COLOR_HAZARD)
            self.screen.blit(warning_text, (self.WIDTH/2 - warning_text.get_width()/2, 10))

        score_text = self.font_small.render(f"SCORE: {self.score:.0f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        dist_text = self.font_small.render(f"CORE DISTANCE: {max(0, int(self.distance_to_core))}", True, self.COLOR_UI_TEXT)
        self.screen.blit(dist_text, (self.WIDTH - dist_text.get_width() - 10, 10))
        
        bar_width = 150
        bar_height = 10
        bar_x = (self.WIDTH - bar_width) / 2
        bar_y = self.HEIGHT - 25
        size_ratio = (self.player_size - self.PLAYER_MIN_SIZE) / (self.PLAYER_MAX_SIZE - self.PLAYER_MIN_SIZE)
        
        pygame.draw.rect(self.screen, (50, 50, 80), (bar_x, bar_y, bar_width, bar_height), border_radius=3)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, (bar_x, bar_y, bar_width * size_ratio, bar_height), border_radius=3)
        size_text = self.font_small.render("SIZE", True, self.COLOR_UI_TEXT)
        self.screen.blit(size_text, (bar_x + bar_width/2 - size_text.get_width()/2, bar_y - 18))

    def _spawn_particle(self, pos, count, color, max_life, speed, radius=2):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * self.np_random.uniform(0.5, 1) * speed
            self.particles.append({
                'pos': pygame.math.Vector2(pos),
                'vel': vel,
                'life': self.np_random.uniform(max_life * 0.5, max_life),
                'max_life': max_life,
                'color': color,
                'radius': radius
            })

# Example usage to run and visualize the game
if __name__ == '__main__':
    # This block will not be executed in the testing environment, but is useful for local development.
    # To run it, you need to unset the dummy video driver.
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Neutron Star Navigator")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        # --- Player Input ---
        keys = pygame.key.get_pressed()
        movement = 0 # None
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Rendering ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("--- RESET ---")
                obs, info = env.reset()
                total_reward = 0

        if terminated or truncated:
            print(f"Episode finished. Total reward: {total_reward:.2f}. Press 'R' to restart.")
            # In a real training loop, you would reset here. For interactive play, we wait for 'R'.
            
        clock.tick(30) # Run at 30 FPS

    pygame.quit()