import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Defend your cities from a missile barrage by raising and lowering tectonic plates to create defensive shields."
    )
    user_guide = (
        "Controls: Use ←→ arrow keys to select a tectonic plate and ↑↓ to raise or lower it, creating a shield to deflect incoming missiles."
    )
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    MAX_STEPS = 1000
    
    # Game parameters
    NUM_PLATES = 8
    PLATE_HEIGHT_MIN = 0
    PLATE_HEIGHT_MAX = 100
    PLATE_ADJUST_SPEED = 4
    PLATE_DEFLECTION_HEIGHT = 50
    
    NUM_CITIES = 8
    CITY_MAX_HEALTH = 100
    CITY_SIZE = 10
    
    MISSILE_SPEED = 2.5
    MISSILE_DAMAGE = 34
    
    INITIAL_MISSILE_SPAWN_PROB = 0.01
    MISSILE_SPAWN_INCREASE = 0.0001 # Per step
    
    WIN_SURVIVAL_RATE = 0.8
    LOSE_SURVIVAL_RATE = 0.2
    
    # Reward structure
    REWARD_WIN = 50.0
    REWARD_LOSE = -50.0
    REWARD_MISSILE_DEFLECTED = 1.0
    REWARD_CITY_SURVIVAL_PER_STEP = 0.01 # Per city

    # Visuals
    COLOR_BG = (15, 20, 40)
    COLOR_PLATE_LOW = (40, 50, 80)
    COLOR_PLATE_HIGH = (120, 140, 200)
    COLOR_PLATE_BORDER = (200, 220, 255)
    COLOR_SELECTED_GLOW = (255, 255, 0)
    
    COLOR_CITY_HEALTHY = (0, 255, 127) # SpringGreen
    COLOR_CITY_DAMAGED = (255, 69, 0) # OrangeRed
    
    COLOR_MISSILE = (255, 0, 255)
    COLOR_TRAJECTORY = (255, 0, 255, 50)
    
    COLOR_TEXT = (240, 240, 240)
    COLOR_TEXT_SHADOW = (20, 20, 20)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # Initialize state variables to avoid attribute errors
        self.plates = []
        self.cities = []
        self.missiles = []
        self.particles = []
        self.selected_plate_idx = 0
        self.missile_spawn_prob = 0.0
        self.total_initial_health = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self._init_plates()
        self._init_cities()
        
        self.missiles = []
        self.particles = []
        self.selected_plate_idx = 0
        self.missile_spawn_prob = self.INITIAL_MISSILE_SPAWN_PROB
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Action Handling ---
        movement = action[0]
        
        # 3=left, 4=right: Select adjacent tectonic plate
        if movement == 3:
            self.selected_plate_idx = (self.selected_plate_idx - 1) % self.NUM_PLATES
        elif movement == 4:
            self.selected_plate_idx = (self.selected_plate_idx + 1) % self.NUM_PLATES
        
        # 1=up, 2=down: Adjust height of currently selected plate
        plate = self.plates[self.selected_plate_idx]
        if movement == 1:
            plate['height'] = min(self.PLATE_HEIGHT_MAX, plate['height'] + self.PLATE_ADJUST_SPEED)
        elif movement == 2:
            plate['height'] = max(self.PLATE_HEIGHT_MIN, plate['height'] - self.PLATE_ADJUST_SPEED)

        # --- Game Logic ---
        self._update_missile_spawner()
        
        reward = 0
        reward += self._update_missiles()
        self._update_particles()
        
        # Continuous reward for city survival
        reward += len([c for c in self.cities if c['health'] > 0]) * self.REWARD_CITY_SURVIVAL_PER_STEP
        
        self.score += reward
        self.steps += 1
        
        # --- Termination Check ---
        terminated = self._check_termination()
        truncated = False # No truncation condition besides termination
        if terminated:
            self.game_over = True
            survival_rate = self._get_city_survival_rate()
            if self.steps >= self.MAX_STEPS and survival_rate >= self.WIN_SURVIVAL_RATE:
                reward += self.REWARD_WIN
            else:
                reward += self.REWARD_LOSE
            self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

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
            "city_survival_rate": self._get_city_survival_rate(),
        }

    # --- Initialization ---
    def _init_plates(self):
        self.plates = []
        plate_width = self.WIDTH // self.NUM_PLATES
        for i in range(self.NUM_PLATES):
            rect = pygame.Rect(i * plate_width, 0, plate_width, self.HEIGHT)
            self.plates.append({'rect': rect, 'height': self.PLATE_HEIGHT_MIN})

    def _init_cities(self):
        self.cities = []
        self.total_initial_health = 0
        city_y_min, city_y_max = self.HEIGHT * 0.8, self.HEIGHT * 0.95
        
        attempts = 0
        while len(self.cities) < self.NUM_CITIES and attempts < 1000:
            attempts += 1
            pos = pygame.math.Vector2(
                self.np_random.uniform(self.CITY_SIZE, self.WIDTH - self.CITY_SIZE),
                self.np_random.uniform(city_y_min, city_y_max)
            )
            
            # Ensure cities don't overlap
            too_close = False
            for city in self.cities:
                if pos.distance_to(city['pos']) < self.CITY_SIZE * 2.5:
                    too_close = True
                    break
            if not too_close:
                self.cities.append({
                    'pos': pos,
                    'health': self.CITY_MAX_HEALTH,
                    'initial_health': self.CITY_MAX_HEALTH
                })
                self.total_initial_health += self.CITY_MAX_HEALTH

    # --- Game Logic Updates ---
    def _update_missile_spawner(self):
        if self.np_random.random() < self.missile_spawn_prob and self.cities:
            self._spawn_missile()
        self.missile_spawn_prob += self.MISSILE_SPAWN_INCREASE

    def _spawn_missile(self):
        healthy_cities = [c for c in self.cities if c['health'] > 0]
        if not healthy_cities: return
        
        target_city = self.np_random.choice(healthy_cities)
        
        start_pos = pygame.math.Vector2(self.np_random.uniform(0, self.WIDTH), 0)
        direction = (target_city['pos'] - start_pos).normalize()
        
        self.missiles.append({
            'pos': start_pos,
            'vel': direction * self.MISSILE_SPEED,
            'target_pos': target_city['pos']
        })

    def _update_missiles(self):
        deflection_reward = 0
        for missile in self.missiles[:]:
            missile['pos'] += missile['vel']
            
            # Check for city collision
            collided = False
            for city in self.cities:
                if city['health'] > 0 and missile['pos'].distance_to(city['pos']) < self.CITY_SIZE:
                    city['health'] = max(0, city['health'] - self.MISSILE_DAMAGE)
                    self._create_explosion(missile['pos'], 30, 50, self.COLOR_CITY_DAMAGED)
                    self.missiles.remove(missile)
                    collided = True
                    break
            if collided:
                continue

            # Check for terrain deflection
            if 0 <= missile['pos'].x < self.WIDTH:
                plate_idx = int(missile['pos'].x // (self.WIDTH / self.NUM_PLATES))
                plate = self.plates[plate_idx]
                if plate['height'] > self.PLATE_DEFLECTION_HEIGHT:
                    if plate['rect'].collidepoint(missile['pos']):
                        deflection_reward += self.REWARD_MISSILE_DEFLECTED
                        self._create_explosion(missile['pos'], 20, 30, self.COLOR_PLATE_HIGH)
                        self.missiles.remove(missile)
                        continue
            
            # Check for out of bounds
            if not self.screen.get_rect().collidepoint(missile['pos']):
                self.missiles.remove(missile)
        
        return deflection_reward

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            p['radius'] *= 0.95
            if p['lifespan'] <= 0 or p['radius'] < 0.5:
                self.particles.remove(p)

    def _check_termination(self):
        return self.steps >= self.MAX_STEPS or self._get_city_survival_rate() < self.LOSE_SURVIVAL_RATE

    # --- Helper Functions ---
    def _get_city_survival_rate(self):
        if self.total_initial_health == 0: return 0.0
        current_health = sum(c['health'] for c in self.cities)
        return current_health / self.total_initial_health

    def _create_explosion(self, pos, radius, count, base_color):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed
            
            r, g, b = base_color
            color_offset = self.np_random.integers(-30, 31)
            
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'lifespan': self.np_random.integers(20, 41),
                'radius': self.np_random.uniform(radius * 0.2, radius * 0.5),
                'color': (
                    min(255, max(0, r + color_offset)),
                    min(255, max(0, g + color_offset)),
                    min(255, max(0, b + color_offset))
                )
            })

    # --- Rendering ---
    def _render_game(self):
        self._render_plates()
        self._render_missiles()
        self._render_cities()
        self._render_particles()

    def _render_plates(self):
        for i, plate in enumerate(self.plates):
            height_ratio = plate['height'] / self.PLATE_HEIGHT_MAX
            color = tuple(int(c1 + (c2 - c1) * height_ratio) for c1, c2 in zip(self.COLOR_PLATE_LOW, self.COLOR_PLATE_HIGH))
            pygame.draw.rect(self.screen, color, plate['rect'])
            pygame.draw.rect(self.screen, self.COLOR_PLATE_BORDER, plate['rect'], 1)

            if i == self.selected_plate_idx:
                # Glow effect
                glow_rect = plate['rect'].inflate(8, 8)
                s = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
                pygame.draw.rect(s, (*self.COLOR_SELECTED_GLOW, 50), s.get_rect(), border_radius=5)
                pygame.draw.rect(s, (*self.COLOR_SELECTED_GLOW, 150), s.get_rect(), width=2, border_radius=5)
                self.screen.blit(s, glow_rect.topleft)

    def _render_cities(self):
        for city in self.cities:
            if city['health'] <= 0:
                # Render as rubble
                pygame.gfxdraw.box(self.screen, 
                                   pygame.Rect(int(city['pos'].x - self.CITY_SIZE / 2), int(city['pos'].y - self.CITY_SIZE / 2), self.CITY_SIZE, self.CITY_SIZE),
                                   (80, 80, 80, 150))
                continue
            
            health_ratio = city['health'] / city['initial_health']
            color = tuple(int(c1 + (c2 - c1) * health_ratio) for c1, c2 in zip(self.COLOR_CITY_DAMAGED, self.COLOR_CITY_HEALTHY))
            
            pos = (int(city['pos'].x), int(city['pos'].y))
            size = int(self.CITY_SIZE / 2)
            points = [(pos[0], pos[1] - size), (pos[0] - size, pos[1]), (pos[0], pos[1] + size), (pos[0] + size, pos[1])]
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)

    def _render_missiles(self):
        for missile in self.missiles:
            # Trajectory line
            start = (int(missile['pos'].x), int(missile['pos'].y))
            end = (int(missile['target_pos'].x), int(missile['target_pos'].y))
            pygame.draw.line(self.screen, self.COLOR_TRAJECTORY, start, end, 1)
            
            # Missile head
            pygame.gfxdraw.aacircle(self.screen, int(missile['pos'].x), int(missile['pos'].y), 3, self.COLOR_MISSILE)
            pygame.gfxdraw.filled_circle(self.screen, int(missile['pos'].x), int(missile['pos'].y), 3, self.COLOR_MISSILE)

    def _render_particles(self):
        for p in self.particles:
            if p['radius'] > 1:
                alpha = int(255 * (p['lifespan'] / 40))
                color = (*p['color'], alpha)
                s = pygame.Surface((p['radius']*2, p['radius']*2), pygame.SRCALPHA)
                pygame.draw.circle(s, color, (p['radius'], p['radius']), p['radius'])
                self.screen.blit(s, (int(p['pos'].x - p['radius']), int(p['pos'].y - p['radius'])), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_ui(self):
        # City Survival Rate
        survival_rate = self._get_city_survival_rate()
        survival_text = f"CITIES: {survival_rate:.0%}"
        self._draw_text(survival_text, (15, 10), self.COLOR_TEXT, self.font_small)
        
        # Timer
        time_left = self.MAX_STEPS - self.steps
        time_text = f"TIME: {time_left}"
        self._draw_text(time_text, (self.WIDTH - 15, 10), self.COLOR_TEXT, self.font_small, align="right")

        # Game Over Message
        if self.game_over:
            survival_rate = self._get_city_survival_rate()
            if self.steps >= self.MAX_STEPS and survival_rate >= self.WIN_SURVIVAL_RATE:
                msg = "MISSION SUCCESS"
                color = self.COLOR_CITY_HEALTHY
            else:
                msg = "MISSION FAILED"
                color = self.COLOR_CITY_DAMAGED
            
            self._draw_text(msg, (self.WIDTH / 2, self.HEIGHT / 2 - 20), color, self.font_large, align="center")
            
    def _draw_text(self, text, pos, color, font, align="left"):
        shadow_surf = font.render(text, True, self.COLOR_TEXT_SHADOW)
        text_surf = font.render(text, True, color)
        
        if align == "center":
            text_rect = text_surf.get_rect(center=pos)
        elif align == "right":
            text_rect = text_surf.get_rect(topright=pos)
        else: # "left"
            text_rect = text_surf.get_rect(topleft=pos)
            
        shadow_rect = text_rect.move(1, 1)
        
        self.screen.blit(shadow_surf, shadow_rect)
        self.screen.blit(text_surf, text_rect)

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # Un-comment the next line to run with a display
    # os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Pygame setup for manual play
    try:
        screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
        pygame.display.set_caption("TerraGuard")
        clock = pygame.time.Clock()
        headless = False
    except pygame.error:
        print("Pygame display unavailable. Running in headless mode.")
        headless = True
        clock = pygame.time.Clock()

    
    movement = 0
    
    print("--- Manual Control ---")
    print(GameEnv.user_guide)
    print("Q: Quit")
    
    while not done:
        action = [0, 0, 0] # Default no-op
        
        if not headless:
            # --- Action Mapping for Manual Play ---
            # 0=none, 1=up, 2=down, 3=left, 4=right
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                movement = 1
            elif keys[pygame.K_DOWN]:
                movement = 2
            elif keys[pygame.K_LEFT]:
                movement = 3
            elif keys[pygame.K_RIGHT]:
                movement = 4
            else:
                movement = 0
                
            action = [movement, 0, 0]

            # --- Event Handling ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                    done = True
        
        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # --- Rendering ---
        if not headless:
            # The observation is already a rendered frame
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
        
        clock.tick(30) # Run at 30 FPS

    env.close()
    print("Game Over. Final Info:", info)