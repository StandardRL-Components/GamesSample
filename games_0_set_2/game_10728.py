import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:50:57.780265
# Source Brief: brief_00728.md
# Brief Index: 728
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math

class GameEnv(gym.Env):
    """
    Navigate a microscopic world as a virus, using momentum-based movement to infect a target cell 
    while crafting boosts from stolen components and evading immune responses.
    """
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Navigate a microscopic world as a virus, using momentum-based movement to infect a target cell "
        "while crafting boosts and evading immune responses."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move. Press space to craft a speed boost and shift to craft a defense boost."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Gymnasium Spaces ---
        self.observation_space = Box(low=0, high=255, shape=(400, 640, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        self.WIDTH, self.HEIGHT = 640, 400
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_big = pygame.font.SysFont("monospace", 48, bold=True)

        # --- Game Constants ---
        self.WORLD_SCALE = 2.5
        self.WORLD_WIDTH = int(self.WIDTH * self.WORLD_SCALE)
        self.WORLD_HEIGHT = int(self.HEIGHT * self.WORLD_SCALE)
        self.MAX_STEPS = 5000
        self.VIRUS_RADIUS = 12
        self.CELL_MIN_RADIUS = 25
        self.CELL_MAX_RADIUS = 45
        self.COMPONENT_COST = 3

        # --- Colors ---
        self.COLOR_BG = (10, 20, 30)
        self.COLOR_BG_PARTICLE = (25, 40, 55)
        self.COLOR_VIRUS = (0, 255, 128)
        self.COLOR_VIRUS_GLOW = (0, 255, 128, 50)
        self.COLOR_TARGET = (255, 50, 100)
        self.COLOR_TARGET_GLOW = (255, 50, 100, 70)
        self.COLOR_CELL = (40, 80, 150)
        self.COLOR_CELL_WALL = (80, 150, 255)
        self.COLOR_COMPONENT = (255, 255, 0)
        self.COLOR_SEEKER = (200, 100, 255)
        self.COLOR_SEEKER_GLOW = (200, 100, 255, 60)
        self.COLOR_SPEED_BOOST = (0, 180, 255)
        self.COLOR_DEFENSE_BOOST = (255, 150, 0)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_TRAIL = (0, 255, 128, 150)

        # --- Physics ---
        self.VIRUS_ACCELERATION = 0.25
        self.VIRUS_MAX_SPEED = 5.0
        self.VIRUS_DRAG = 0.98
        self.SEEKER_BASE_SPEED = 1.0
        
        # --- Initialize State ---
        self.virus = {}
        self.cells = []
        self.target_cell = {}
        self.seekers = []
        self.particles = []
        self.bg_particles = []
        self.prev_action = [0, 0, 0]
        self.steps = 0
        self.score = 0
        self.game_over_reason = ""

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # --- Reset Game State ---
        self.steps = 0
        self.score = 0
        self.game_over_reason = ""
        self.prev_action = [0, 0, 0]
        
        # --- Level Generation ---
        self._generate_level()
        self.distance_to_target = self._get_distance(self.virus['pos'], self.target_cell['pos'])

        # --- Reset Entity Lists ---
        self.seekers.clear()
        self.particles.clear()
        
        # --- Background Ambiance ---
        self.bg_particles = [{
            'pos': pygame.Vector2(self.np_random.uniform(0, self.WORLD_WIDTH), self.np_random.uniform(0, self.WORLD_HEIGHT)),
            'radius': self.np_random.uniform(1, 3),
            'drift': pygame.Vector2(self.np_random.uniform(-0.1, 0.1), self.np_random.uniform(-0.1, 0.1))
        } for _ in range(150)]
        
        return self._get_observation(), self._get_info()

    def _generate_level(self):
        # Place virus near center-left
        self.virus = {
            'pos': pygame.Vector2(self.WORLD_WIDTH * 0.2, self.WORLD_HEIGHT * 0.5),
            'vel': pygame.Vector2(0, 0),
            'radius': self.VIRUS_RADIUS,
            'components': 0,
            'speed_boost_timer': 0,
            'defense_boost_timer': 0,
        }

        # Place target cell near center-right
        self.target_cell = {
            'pos': pygame.Vector2(self.WORLD_WIDTH * 0.8, self.WORLD_HEIGHT * 0.5),
            'radius': self.CELL_MAX_RADIUS * 1.5,
        }

        # Generate cells
        self.cells = []
        num_cells = int((self.WORLD_WIDTH * self.WORLD_HEIGHT) / (self.CELL_MAX_RADIUS**2 * np.pi * 5))
        for _ in range(num_cells):
            radius = self.np_random.uniform(self.CELL_MIN_RADIUS, self.CELL_MAX_RADIUS)
            pos = pygame.Vector2(
                self.np_random.uniform(radius, self.WORLD_WIDTH - radius),
                self.np_random.uniform(radius, self.WORLD_HEIGHT - radius)
            )
            
            # Avoid placing cells too close to start or end
            if self._get_distance(pos, self.virus['pos']) < 200 or self._get_distance(pos, self.target_cell['pos']) < 200:
                continue

            # Check for overlap with other cells
            is_overlapping = False
            for other_cell in self.cells:
                if self._get_distance(pos, other_cell['pos']) < radius + other_cell['radius'] + 20:
                    is_overlapping = True
                    break
            if not is_overlapping:
                has_component = self.np_random.random() < 0.4
                self.cells.append({'pos': pos, 'radius': radius, 'has_component': has_component})

    def step(self, action):
        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        
        reward = 0
        terminated = False
        truncated = False

        # --- 1. Handle Player Actions ---
        # Movement
        force = pygame.Vector2(0, 0)
        if movement == 1: force.y = -1
        elif movement == 2: force.y = 1
        elif movement == 3: force.x = -1
        elif movement == 4: force.x = 1
        if force.length() > 0:
            force.normalize_ip()
            
        acceleration_factor = self.VIRUS_ACCELERATION * (2 if self.virus['speed_boost_timer'] > 0 else 1)
        self.virus['vel'] += force * acceleration_factor
        
        # Crafting (edge-triggered)
        if space_pressed and not self.prev_action[1]: # Craft speed boost
            if self.virus['components'] >= self.COMPONENT_COST:
                self.virus['components'] -= self.COMPONENT_COST
                self.virus['speed_boost_timer'] = 300 # 10 seconds at 30fps
                reward += 1.0 # Reward for crafting
                self._create_effect(self.virus['pos'], self.COLOR_SPEED_BOOST, 20)
        
        if shift_pressed and not self.prev_action[2]: # Craft defense boost
            if self.virus['components'] >= self.COMPONENT_COST:
                self.virus['components'] -= self.COMPONENT_COST
                self.virus['defense_boost_timer'] = 450 # 15 seconds at 30fps
                reward += 1.0 # Reward for crafting
                self._create_effect(self.virus['pos'], self.COLOR_DEFENSE_BOOST, 20)

        self.prev_action = [movement, space_pressed, shift_pressed]

        # --- 2. Update Game State ---
        # Update virus
        if self.virus['vel'].length() > self.VIRUS_MAX_SPEED:
            self.virus['vel'].scale_to_length(self.VIRUS_MAX_SPEED)
        self.virus['pos'] += self.virus['vel']
        self.virus['vel'] *= self.VIRUS_DRAG
        
        # Update boost timers
        self.virus['speed_boost_timer'] = max(0, self.virus['speed_boost_timer'] - 1)
        self.virus['defense_boost_timer'] = max(0, self.virus['defense_boost_timer'] - 1)
        
        # Keep virus in bounds
        self.virus['pos'].x = np.clip(self.virus['pos'].x, self.virus['radius'], self.WORLD_WIDTH - self.virus['radius'])
        self.virus['pos'].y = np.clip(self.virus['pos'].y, self.virus['radius'], self.WORLD_HEIGHT - self.virus['radius'])

        # Add player trail
        if self.virus['vel'].length() > 0.5:
            self.particles.append(self._create_trail_particle(self.virus['pos']))

        # Update immune system
        self._update_seekers(reward)
        self._spawn_seekers()
        
        # Update particles
        self._update_particles()

        # --- 3. Check Collisions & Events ---
        # Virus vs Cells (for components)
        for cell in self.cells:
            if cell['has_component']:
                dist = self._get_distance(self.virus['pos'], cell['pos'])
                if dist < self.virus['radius'] + cell['radius']:
                    cell['has_component'] = False
                    self.virus['components'] += 1
                    reward += 0.1 # Reward for component
                    self.score += 10
                    self._create_effect(cell['pos'], self.COLOR_COMPONENT, 15, num_particles=5)

        # Virus vs Seekers
        seekers_to_remove = []
        for seeker in self.seekers:
            dist = self._get_distance(self.virus['pos'], seeker['pos'])
            if dist < self.virus['radius'] + seeker['radius']:
                if self.virus['defense_boost_timer'] > 0:
                    self.virus['defense_boost_timer'] = 0 # Shield breaks
                    seekers_to_remove.append(seeker)
                    reward += 2.0 # Reward for using shield
                    self.score += 50
                    self._create_effect(self.virus['pos'], self.COLOR_DEFENSE_BOOST, 30, num_particles=25)
                else:
                    terminated = True
                    reward = -100.0
                    self.score = max(0, self.score - 100)
                    self.game_over_reason = "DESTROYED"
                    self._create_effect(self.virus['pos'], self.COLOR_VIRUS, 50, num_particles=50)
                    break
        if terminated:
            self.seekers = [s for s in self.seekers if s not in seekers_to_remove]
            return self._get_observation(), reward, terminated, truncated, self._get_info()
        self.seekers = [s for s in self.seekers if s not in seekers_to_remove]

        # Virus vs Target Cell
        dist = self._get_distance(self.virus['pos'], self.target_cell['pos'])
        if dist < self.virus['radius'] + self.target_cell['radius']:
            terminated = True
            reward = 100.0
            self.score += 1000
            self.game_over_reason = "TARGET INFECTED"
            self._create_effect(self.target_cell['pos'], self.COLOR_TARGET, 100, num_particles=100)
        
        # --- 4. Calculate Final Reward & Check Termination ---
        self.steps += 1
        
        # Proximity reward
        new_dist_to_target = self._get_distance(self.virus['pos'], self.target_cell['pos'])
        reward += (self.distance_to_target - new_dist_to_target) * 0.01
        self.distance_to_target = new_dist_to_target
        
        if self.steps >= self.MAX_STEPS:
            terminated = True
            truncated = True
            if not self.game_over_reason:
                self.game_over_reason = "TIME LIMIT REACHED"
                reward -= 10.0 # Penalty for running out of time
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _update_seekers(self, reward_ref):
        seekers_to_remove = []
        closest_seeker_dist = float('inf')
        for seeker in self.seekers:
            dir_to_virus = (self.virus['pos'] - seeker['pos'])
            dist = dir_to_virus.length()
            closest_seeker_dist = min(closest_seeker_dist, dist)
            
            if dist > 1: # Avoid division by zero
                dir_to_virus.normalize_ip()
            
            speed_increase = (self.steps / 100) * 0.01
            current_speed = self.SEEKER_BASE_SPEED + speed_increase
            seeker['vel'] = seeker['vel'] * 0.9 + dir_to_virus * current_speed * 0.1
            seeker['pos'] += seeker['vel']
            
            # Trail
            if self.np_random.random() < 0.2:
                 self.particles.append(self._create_trail_particle(seeker['pos'], self.COLOR_SEEKER, 0.5))

        # Close call reward
        if 50 < closest_seeker_dist < 70:
            reward_ref += 0.05 # Small bonus for kiting
        
    def _spawn_seekers(self):
        spawn_rate = min(0.1, 0.001 + (self.steps * 0.00005))
        if self.np_random.random() < spawn_rate:
            # Spawn off-screen
            cam_rect = self._get_camera_rect()
            edge = self.np_random.choice(['top', 'bottom', 'left', 'right'])
            if edge == 'top':
                pos = pygame.Vector2(self.np_random.uniform(cam_rect.left, cam_rect.right), cam_rect.top - 20)
            elif edge == 'bottom':
                pos = pygame.Vector2(self.np_random.uniform(cam_rect.left, cam_rect.right), cam_rect.bottom + 20)
            elif edge == 'left':
                pos = pygame.Vector2(cam_rect.left - 20, self.np_random.uniform(cam_rect.top, cam_rect.bottom))
            elif edge == 'right':
                pos = pygame.Vector2(cam_rect.right + 20, self.np_random.uniform(cam_rect.top, cam_rect.bottom))
            
            pos.x = np.clip(pos.x, 0, self.WORLD_WIDTH)
            pos.y = np.clip(pos.y, 0, self.WORLD_HEIGHT)

            self.seekers.append({
                'pos': pos,
                'vel': pygame.Vector2(0,0),
                'radius': 8,
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        cam_x, cam_y = self._get_camera_offset()
        
        self._render_background(cam_x, cam_y)
        self._render_game(cam_x, cam_y)
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_camera_offset(self):
        cam_x = self.virus['pos'].x - self.WIDTH / 2
        cam_y = self.virus['pos'].y - self.HEIGHT / 2
        cam_x = np.clip(cam_x, 0, self.WORLD_WIDTH - self.WIDTH)
        cam_y = np.clip(cam_y, 0, self.WORLD_HEIGHT - self.HEIGHT)
        return cam_x, cam_y

    def _get_camera_rect(self):
        cam_x, cam_y = self._get_camera_offset()
        return pygame.Rect(cam_x, cam_y, self.WIDTH, self.HEIGHT)

    def _render_background(self, cam_x, cam_y):
        for p in self.bg_particles:
            p['pos'] += p['drift']
            if p['pos'].x < 0 or p['pos'].x > self.WORLD_WIDTH: p['drift'].x *= -1
            if p['pos'].y < 0 or p['pos'].y > self.WORLD_HEIGHT: p['drift'].y *= -1
            
            screen_pos = p['pos'] - pygame.Vector2(cam_x, cam_y)
            pygame.gfxdraw.filled_circle(
                self.screen, int(screen_pos.x), int(screen_pos.y),
                int(p['radius']), self.COLOR_BG_PARTICLE
            )

    def _render_game(self, cam_x, cam_y):
        # Render particles
        for p in self.particles:
            screen_pos = p['pos'] - pygame.Vector2(cam_x, cam_y)
            if p['radius'] > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(screen_pos.x), int(screen_pos.y), int(p['radius']), p['color'])

        # Render cells
        for cell in self.cells:
            screen_pos = cell['pos'] - pygame.Vector2(cam_x, cam_y)
            pygame.gfxdraw.aacircle(self.screen, int(screen_pos.x), int(screen_pos.y), int(cell['radius']), self.COLOR_CELL_WALL)
            pygame.gfxdraw.filled_circle(self.screen, int(screen_pos.x), int(screen_pos.y), int(cell['radius']), self.COLOR_CELL)
            if cell['has_component']:
                pygame.gfxdraw.filled_circle(self.screen, int(screen_pos.x), int(screen_pos.y), int(cell['radius'] * 0.4), self.COLOR_COMPONENT)
                pygame.gfxdraw.aacircle(self.screen, int(screen_pos.x), int(screen_pos.y), int(cell['radius'] * 0.4), (255,255,255))

        # Render target cell
        pulse = (math.sin(self.steps * 0.1) + 1) / 2
        target_radius = self.target_cell['radius'] + pulse * 5
        screen_pos = self.target_cell['pos'] - pygame.Vector2(cam_x, cam_y)
        pygame.gfxdraw.filled_circle(self.screen, int(screen_pos.x), int(screen_pos.y), int(target_radius + 10), self.COLOR_TARGET_GLOW)
        pygame.gfxdraw.filled_circle(self.screen, int(screen_pos.x), int(screen_pos.y), int(target_radius), self.COLOR_TARGET)
        pygame.gfxdraw.aacircle(self.screen, int(screen_pos.x), int(screen_pos.y), int(target_radius), (255,255,255))

        # Render seekers
        for seeker in self.seekers:
            screen_pos = seeker['pos'] - pygame.Vector2(cam_x, cam_y)
            pygame.gfxdraw.filled_circle(self.screen, int(screen_pos.x), int(screen_pos.y), int(seeker['radius'] + 5), self.COLOR_SEEKER_GLOW)
            pygame.gfxdraw.filled_circle(self.screen, int(screen_pos.x), int(screen_pos.y), int(seeker['radius']), self.COLOR_SEEKER)

        # Render virus
        if self.game_over_reason not in ["DESTROYED", "TARGET INFECTED"]:
            screen_pos = self.virus['pos'] - pygame.Vector2(cam_x, cam_y)
            
            # Glow
            glow_radius = self.virus['radius'] + 10 + (math.sin(self.steps * 0.2) * 3)
            pygame.gfxdraw.filled_circle(self.screen, int(screen_pos.x), int(screen_pos.y), int(glow_radius), self.COLOR_VIRUS_GLOW)
            
            # Boost auras
            if self.virus['speed_boost_timer'] > 0:
                aura_alpha = int(100 * (self.virus['speed_boost_timer'] / 300))
                pygame.gfxdraw.filled_circle(self.screen, int(screen_pos.x), int(screen_pos.y), int(self.virus['radius'] + 8), (*self.COLOR_SPEED_BOOST, aura_alpha))
            if self.virus['defense_boost_timer'] > 0:
                shield_alpha = int(150 * (self.virus['defense_boost_timer'] / 450))
                pygame.gfxdraw.aacircle(self.screen, int(screen_pos.x), int(screen_pos.y), int(self.virus['radius'] + 5), (*self.COLOR_DEFENSE_BOOST, shield_alpha))
                pygame.gfxdraw.aacircle(self.screen, int(screen_pos.x), int(screen_pos.y), int(self.virus['radius'] + 6), (*self.COLOR_DEFENSE_BOOST, shield_alpha))

            # Core
            pygame.gfxdraw.filled_circle(self.screen, int(screen_pos.x), int(screen_pos.y), self.virus['radius'], self.COLOR_VIRUS)
            pygame.gfxdraw.aacircle(self.screen, int(screen_pos.x), int(screen_pos.y), self.virus['radius'], (200,255,220))

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Components
        comp_text = self.font_main.render(f"COMPONENTS: {self.virus['components']}", True, self.COLOR_COMPONENT)
        self.screen.blit(comp_text, (self.WIDTH // 2 - comp_text.get_width() // 2, 10))

        # Boosts
        speed_text = self.font_main.render(f"SPD: {self.virus['speed_boost_timer']//30}s", True, self.COLOR_SPEED_BOOST if self.virus['speed_boost_timer'] > 0 else self.COLOR_TEXT)
        self.screen.blit(speed_text, (self.WIDTH - 180, 10))
        def_text = self.font_main.render(f"DEF: {self.virus['defense_boost_timer']//30}s", True, self.COLOR_DEFENSE_BOOST if self.virus['defense_boost_timer'] > 0 else self.COLOR_TEXT)
        self.screen.blit(def_text, (self.WIDTH - 90, 10))

        # Game Over Message
        if self.game_over_reason:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            end_text = self.font_big.render(self.game_over_reason, True, self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "components": self.virus['components']}

    # --- Helper Functions ---
    def _get_distance(self, pos1, pos2):
        return math.sqrt((pos1.x - pos2.x)**2 + (pos1.y - pos2.y)**2)

    def _create_effect(self, pos, color, radius, num_particles=20):
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'radius': self.np_random.uniform(radius/4, radius/2),
                'decay': self.np_random.uniform(0.02, 0.05),
                'color': (*color, 255)
            })

    def _create_trail_particle(self, pos, color=None, scale=1.0):
        if color is None: color = self.COLOR_TRAIL
        return {
            'pos': pos.copy() + pygame.Vector2(self.np_random.uniform(-3,3), self.np_random.uniform(-3,3)),
            'vel': pygame.Vector2(0,0),
            'radius': self.np_random.uniform(1, 4) * scale,
            'decay': self.np_random.uniform(0.05, 0.1),
            'color': color
        }

    def _update_particles(self):
        particles_to_keep = []
        for p in self.particles:
            p['pos'] += p['vel']
            p['radius'] -= p['decay'] * p['radius']
            if p['radius'] > 0.5:
                particles_to_keep.append(p)
        self.particles = particles_to_keep

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block is for human play and is not used by the evaluation system.
    # It is not necessary to modify this block.
    try:
        env = GameEnv()
        obs, info = env.reset()
        
        # Un-dummy the video driver for human play
        os.environ["SDL_VIDEODRIVER"] = "x11" 
        pygame.display.init()
        pygame.font.init()

        pygame.display.set_caption("Microbial Mayhem")
        screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        clock = pygame.time.Clock()
        
        running = True
        total_reward = 0
        
        while running:
            action = [0, 0, 0] # Default no-op
            keys = pygame.key.get_pressed()
            
            if keys[pygame.K_UP]: action[0] = 1
            elif keys[pygame.K_DOWN]: action[0] = 2
            elif keys[pygame.K_LEFT]: action[0] = 3
            elif keys[pygame.K_RIGHT]: action[0] = 4
            
            if keys[pygame.K_SPACE]: action[1] = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            # Draw the observation from the environment
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            
            pygame.display.flip()
            
            if terminated or truncated:
                print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
                pygame.time.wait(2000)
                obs, info = env.reset()
                total_reward = 0
                
            clock.tick(30) # Run at 30 FPS

    finally:
        env.close()