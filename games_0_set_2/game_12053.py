import gymnasium as gym
import os
import pygame
import numpy as np
from gymnasium.spaces import MultiDiscrete, Box
import pygame.gfxdraw
import math
import random
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Helper classes for game entities
class Particle:
    def __init__(self, x, y, color, life, dx, dy, radius, gravity=0.0):
        self.x = x
        self.y = y
        self.color = color
        self.life = life
        self.max_life = life
        self.dx = dx
        self.dy = dy
        self.radius = radius
        self.gravity = gravity

    def update(self):
        self.x += self.dx
        self.y += self.dy
        self.dy += self.gravity
        self.life -= 1

    def draw(self, surface):
        if self.life > 0:
            alpha = int(255 * (self.life / self.max_life))
            r = int(self.radius * (self.life / self.max_life))
            if r > 0:
                pygame.gfxdraw.filled_circle(surface, int(self.x), int(self.y), r, (*self.color, alpha))

class Collector:
    def __init__(self, start_pos, vel, gas_types):
        self.pos = np.array(start_pos, dtype=float)
        self.vel = np.array(vel, dtype=float)
        self.start_pos = np.array(start_pos, dtype=float)
        self.state = "LAUNCHED"  # LAUNCHED, COLLECTING, RETURNING
        self.capacity = 100
        self.contents = {gas: 0 for gas in gas_types}
        self.total_content = 0
        self.collect_timer = 0
        self.gravity = 0.05
        self.path = [tuple(self.pos.astype(int))]

    def update(self):
        if self.state == "LAUNCHED":
            self.vel[1] += self.gravity
            self.pos += self.vel
            self.path.append(tuple(self.pos.astype(int)))
            if len(self.path) > 100: self.path.pop(0)
            if self.pos[1] > 400 or self.pos[0] < 0 or self.pos[0] > 640:
                self.state = "RETURNING"
        
        elif self.state == "COLLECTING":
            self.collect_timer -= 1
            if self.collect_timer <= 0 or self.total_content >= self.capacity:
                self.state = "RETURNING"
        
        elif self.state == "RETURNING":
            direction = self.start_pos - self.pos
            dist = np.linalg.norm(direction)
            if dist < 5:
                return "ARRIVED"
            self.pos += (direction / dist) * 5 # Return speed
        
        return None

    def collect_gas(self, gas_type, amount):
        if self.total_content < self.capacity:
            collect_amount = min(amount, self.capacity - self.total_content)
            self.contents[gas_type] += collect_amount
            self.total_content += collect_amount
            return collect_amount
        return 0

    def draw(self, surface):
        # Draw path
        if len(self.path) > 2 and self.state == "LAUNCHED":
            pygame.draw.aalines(surface, (255, 255, 255, 50), False, self.path)
        
        # Draw collector
        color = (200, 200, 255) if self.state != "COLLECTING" else (255, 255, 100)
        pos_int = self.pos.astype(int)
        
        # Glow
        pygame.gfxdraw.filled_circle(surface, pos_int[0], pos_int[1], 10, (*color, 50))
        pygame.gfxdraw.filled_circle(surface, pos_int[0], pos_int[1], 7, color)
        
        # Fill indicator
        if self.total_content > 0:
            fill_ratio = self.total_content / self.capacity
            fill_color = (100, 255, 100)
            pygame.draw.arc(surface, fill_color, (pos_int[0]-6, pos_int[1]-6, 12, 12), 0, fill_ratio * 2 * math.pi, 2)


class Cloud:
    def __init__(self, x, y, radius, gas_type, color, drift_speed):
        self.x = x
        self.y = y
        self.radius = radius
        self.gas_type = gas_type
        self.color = color
        self.drift = np.array([random.uniform(-1, 1), random.uniform(-1, 1)])
        self.drift = self.drift / np.linalg.norm(self.drift) * drift_speed
        self.gas_amount = 1000

    def update(self, bounds):
        self.x += self.drift[0]
        self.y += self.drift[1]
        
        if self.x - self.radius > bounds[0] or self.x + self.radius < 0 or \
           self.y - self.radius > bounds[1] or self.y + self.radius < 0:
            return True # Needs respawn
            
        self.gas_amount = min(1000, self.gas_amount + 0.1) # Regenerate gas
        return False

    def draw(self, surface):
        # Draw multiple transparent circles for a nebula effect
        for i in range(5):
            offset_x = random.uniform(-0.3, 0.3) * self.radius
            offset_y = random.uniform(-0.3, 0.3) * self.radius
            r = int(self.radius * random.uniform(0.5, 1.0) * (self.gas_amount / 1000))
            if r > 0:
                alpha = int(random.randint(15, 30) * (self.gas_amount / 1000))
                pygame.gfxdraw.filled_circle(surface, int(self.x + offset_x), int(self.y + offset_y), r, (*self.color, alpha))

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Launch collector drones to harvest gas from drifting nebulae. Synthesize the "
        "collected resources into fuel to jump between galaxies and explore the cosmos."
    )
    user_guide = (
        "Use the arrow keys (↑↓←→) to aim your launcher. Press space to launch a collector "
        "drone and press shift to synthesize fuel from collected gas."
    )
    auto_advance = True

    # --- CONSTANTS ---
    WIDTH, HEIGHT = 640, 400
    MAX_STEPS = 5000
    TARGET_GALAXY = 10
    
    COLOR_BG = (10, 15, 30)
    COLOR_UI_TEXT = (200, 200, 220)
    COLOR_UI_BAR = (40, 60, 100)
    COLOR_FUEL = (100, 255, 100)
    COLOR_ENERGY = (100, 200, 255)
    
    GAS_TYPES = {
        'H': {'name': 'Hydrogen', 'color': (100, 150, 255)},
        'O': {'name': 'Oxygen', 'color': (255, 100, 100)},
        'N': {'name': 'Nitrogen', 'color': (200, 100, 255)},
        'He': {'name': 'Helium', 'color': (255, 255, 100)},
    }
    
    RECIPES = [
        {'req': {'H': 20, 'O': 10}, 'fuel': 50},
        {'req': {'H': 10, 'N': 20}, 'fuel': 75},
        {'req': {'He': 30}, 'fuel': 100},
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(low=0, high=255, shape=(400, 640, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("consolas", 14, bold=True)
        self.font_large = pygame.font.SysFont("consolas", 24, bold=True)
        
        self.render_mode = render_mode
        self.stars = [(random.randint(0, self.WIDTH), random.randint(0, self.HEIGHT), random.randint(1, 2)) for _ in range(150)]
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        self.galaxy_index = 0
        self.synthesizer_speed_multiplier = 1.0
        self.nebula_drift_speed = 0.1
        
        self.unlocked_gas_types = ['H', 'O']
        self.unlocked_recipes_idx = 1
        
        self.max_fuel = 200
        self.fuel_level = self.max_fuel / 4
        self.max_launch_energy = 100
        self.launch_energy = self.max_launch_energy
        
        self.stored_gas = {gas: 0 for gas in self.GAS_TYPES}
        self.synthesizer_pos = (self.WIDTH // 2, self.HEIGHT - 20)
        
        self.aim_cursor_pos = np.array([self.WIDTH / 2, self.HEIGHT - 50], dtype=float)
        self.collectors = []
        self.particles = []
        self.nebulae = []
        self._generate_nebulae()

        self.last_space_held = False
        self.last_shift_held = False
        self.fuel_out_timer = 0
        self.jump_animation_timer = 0
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        self.steps += 1
        
        # Handle jump animation state
        if self.jump_animation_timer > 0:
            self.jump_animation_timer -= 1
            if self.jump_animation_timer == 0:
                self._complete_jump()
            return self._get_observation(), 0, False, False, self._get_info()

        # Handle player input
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_press = space_held and not self.last_space_held
        shift_press = shift_held and not self.last_shift_held
        
        self._update_cursor(movement)
        if space_press: self._launch_collector()
        if shift_press: reward += self._synthesize_fuel()
        
        self.last_space_held, self.last_shift_held = space_held, shift_held

        # Update game world
        self._update_nebulae()
        reward += self._update_collectors()
        self._update_particles()
        self._recharge_energy()
        
        # Check for auto-jump and termination
        jump_reward = self._check_for_galaxy_jump()
        reward += jump_reward
        if jump_reward > 0: self.fuel_out_timer = 0

        terminated, term_reward = self._check_termination()
        reward += term_reward
        self.score += reward

        return self._get_observation(), reward, terminated, False, self._get_info()

    # --- UPDATE LOGIC ---
    def _update_cursor(self, movement):
        move_speed = 4
        if movement == 1: self.aim_cursor_pos[1] -= move_speed
        elif movement == 2: self.aim_cursor_pos[1] += move_speed
        elif movement == 3: self.aim_cursor_pos[0] -= move_speed
        elif movement == 4: self.aim_cursor_pos[0] += move_speed
        
        self.aim_cursor_pos[0] = np.clip(self.aim_cursor_pos[0], 0, self.WIDTH)
        self.aim_cursor_pos[1] = np.clip(self.aim_cursor_pos[1], 0, self.HEIGHT - 40)

    def _launch_collector(self):
        launch_cost = 30
        if self.launch_energy >= launch_cost and len(self.collectors) < 5:
            self.launch_energy -= launch_cost
            direction = self.aim_cursor_pos - np.array(self.synthesizer_pos)
            vel = (direction / np.linalg.norm(direction)) * 8
            
            new_collector = Collector(self.synthesizer_pos, vel, self.GAS_TYPES.keys())
            self.collectors.append(new_collector)
            
            for _ in range(20):
                p_vel_x = -vel[0] * 0.1 + random.uniform(-0.5, 0.5)
                p_vel_y = -vel[1] * 0.1 + random.uniform(-0.5, 0.5)
                self.particles.append(Particle(self.synthesizer_pos[0], self.synthesizer_pos[1]-10, (255,200,150), 30, p_vel_x, p_vel_y, 3))

    def _synthesize_fuel(self):
        reward = 0
        # Iterate recipes from most complex to simplest
        for i in range(self.unlocked_recipes_idx -1, -1, -1):
            recipe = self.RECIPES[i]
            can_craft = True
            for gas, amount in recipe['req'].items():
                if self.stored_gas[gas] < amount:
                    can_craft = False
                    break
            
            if can_craft:
                for gas, amount in recipe['req'].items():
                    self.stored_gas[gas] -= amount
                
                self.fuel_level = min(self.max_fuel, self.fuel_level + recipe['fuel'])
                reward = 2.0

                # Create particles for visual feedback
                for _ in range(50):
                    angle = random.uniform(0, 2*math.pi)
                    speed = random.uniform(1, 3)
                    dx, dy = math.cos(angle)*speed, math.sin(angle)*speed
                    color = self.COLOR_FUEL
                    self.particles.append(Particle(self.synthesizer_pos[0], self.synthesizer_pos[1], color, 40, dx, -abs(dy), 4))
                
                return reward
        return reward

    def _update_collectors(self):
        reward = 0
        for collector in self.collectors[:]:
            status = collector.update()
            if status == "ARRIVED":
                for gas, amount in collector.contents.items():
                    if amount > 0:
                        self.stored_gas[gas] += amount
                if collector.total_content >= collector.capacity:
                    reward += 0.5 # Filled collector bonus
                self.collectors.remove(collector)
                continue

            if collector.state == "LAUNCHED":
                for cloud in self.nebulae:
                    dist = np.linalg.norm(collector.pos - np.array([cloud.x, cloud.y]))
                    if dist < cloud.radius and cloud.gas_amount > 1:
                        collector.state = "COLLECTING"
                        collector.collect_timer = 60 * self.synthesizer_speed_multiplier
                        break
            
            if collector.state == "COLLECTING":
                for cloud in self.nebulae:
                    dist = np.linalg.norm(collector.pos - np.array([cloud.x, cloud.y]))
                    if dist < cloud.radius and cloud.gas_type in self.unlocked_gas_types:
                        if cloud.gas_amount > 0:
                            collected = collector.collect_gas(cloud.gas_type, 2)
                            if collected > 0:
                                cloud.gas_amount -= collected
                                reward += 0.1 # Gas collection reward
        return reward

    def _update_nebulae(self):
        for cloud in self.nebulae[:]:
            if cloud.update((self.WIDTH, self.HEIGHT-100)):
                self.nebulae.remove(cloud)
        
        while len(self.nebulae) < 5 + self.galaxy_index:
            self._spawn_cloud()

    def _update_particles(self):
        self.particles = [p for p in self.particles if p.life > 0]
        for p in self.particles:
            p.update()

    def _recharge_energy(self):
        self.launch_energy = min(self.max_launch_energy, self.launch_energy + 0.2)

    def _check_for_galaxy_jump(self):
        if self.fuel_level >= self.max_fuel:
            self.jump_animation_timer = 60 # 2 seconds at 30fps
            self.galaxy_index += 1
            return 5.0
        return 0.0
    
    def _complete_jump(self):
        self.fuel_level = self.max_fuel / 4
        self.collectors.clear()
        self.particles.clear()
        
        # Difficulty progression
        if self.galaxy_index % 3 == 0 and self.galaxy_index > 0:
            self.synthesizer_speed_multiplier *= 0.95
        if self.galaxy_index % 2 == 0 and self.galaxy_index > 0:
            self.nebula_drift_speed += 0.02
        
        # Unlock new content
        if self.galaxy_index == 2 and 'N' not in self.unlocked_gas_types:
            self.unlocked_gas_types.append('N')
            self.unlocked_recipes_idx = 2
        if self.galaxy_index == 5 and 'He' not in self.unlocked_gas_types:
            self.unlocked_gas_types.append('He')
            self.unlocked_recipes_idx = 3

        self._generate_nebulae()

    def _check_termination(self):
        if self.galaxy_index >= self.TARGET_GALAXY:
            self.game_over = True
            self.win = True
            return True, 100.0

        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True, 0

        can_recover = any(c.total_content > 0 for c in self.collectors) or \
                      any(self.stored_gas[g] > 0 for g in self.unlocked_gas_types)

        if self.fuel_level <= 0 and not can_recover:
            self.fuel_out_timer += 1
        else:
            self.fuel_out_timer = 0
            
        if self.fuel_out_timer > 100:
            self.game_over = True
            return True, -100.0
            
        return False, 0

    # --- GENERATION ---
    def _generate_nebulae(self):
        self.nebulae.clear()
        for _ in range(5 + self.galaxy_index):
            self._spawn_cloud()

    def _spawn_cloud(self):
        gas_type = random.choice(self.unlocked_gas_types)
        props = self.GAS_TYPES[gas_type]
        cloud = Cloud(
            x=random.randint(50, self.WIDTH - 50),
            y=random.randint(50, self.HEIGHT - 150),
            radius=random.randint(30, 60),
            gas_type=gas_type,
            color=props['color'],
            drift_speed=self.nebula_drift_speed
        )
        self.nebulae.append(cloud)

    # --- GYM INTERFACE ---
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_game()
        self._render_ui()
        
        if self.jump_animation_timer > 0:
            self._render_jump_animation()

        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "galaxy": self.galaxy_index,
            "fuel": self.fuel_level,
            "stored_gas": self.stored_gas,
        }

    # --- RENDERING ---
    def _render_background(self):
        for x, y, size in self.stars:
            self.screen.set_at((x, y), (200, 200, 200, 100))

    def _render_game(self):
        for cloud in self.nebulae: cloud.draw(self.screen)
        self._render_synthesizer()
        for collector in self.collectors: collector.draw(self.screen)
        for p in self.particles: p.draw(self.screen)
        self._render_aim_cursor()

    def _render_synthesizer(self):
        pos = self.synthesizer_pos
        pygame.gfxdraw.filled_polygon(self.screen, [(pos[0]-40, pos[1]+20), (pos[0]+40, pos[1]+20), (pos[0]+20, pos[1]-10), (pos[0]-20, pos[1]-10)], (60,70,90))
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1]-10, 15, (150,180,255))

    def _render_aim_cursor(self):
        pos = self.aim_cursor_pos.astype(int)
        color = (255, 100, 100)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 10, color)
        pygame.draw.line(self.screen, color, (pos[0]-15, pos[1]), (pos[0]+15, pos[1]), 1)
        pygame.draw.line(self.screen, color, (pos[0], pos[1]-15), (pos[0], pos[1]+15), 1)

    def _render_ui(self):
        # Fuel Bar
        bar_w = 200
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR, (10, 10, bar_w, 20))
        fuel_w = int(bar_w * (self.fuel_level / self.max_fuel))
        pygame.draw.rect(self.screen, self.COLOR_FUEL, (10, 10, fuel_w, 20))
        fuel_text = self.font_small.render("FUEL", True, self.COLOR_UI_TEXT)
        self.screen.blit(fuel_text, (15, 12))
        
        # Launch Energy Bar
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR, (10, 35, bar_w, 10))
        energy_w = int(bar_w * (self.launch_energy / self.max_launch_energy))
        pygame.draw.rect(self.screen, self.COLOR_ENERGY, (10, 35, energy_w, 10))
        
        # Galaxy Info
        galaxy_text = self.font_large.render(f"Galaxy: {self.galaxy_index}/{self.TARGET_GALAXY}", True, self.COLOR_UI_TEXT)
        self.screen.blit(galaxy_text, (self.WIDTH - galaxy_text.get_width() - 10, 10))
        
        # Stored Gas Info
        y_offset = 0
        for gas_type in self.unlocked_gas_types:
            props = self.GAS_TYPES[gas_type]
            gas_amount = int(self.stored_gas[gas_type])
            
            pygame.draw.rect(self.screen, props['color'], (15, self.HEIGHT - 80 + y_offset, 10, 10))
            text = self.font_small.render(f"{props['name']}: {gas_amount}", True, self.COLOR_UI_TEXT)
            self.screen.blit(text, (30, self.HEIGHT - 82 + y_offset))
            y_offset += 20
            
    def _render_jump_animation(self):
        progress = (60 - self.jump_animation_timer) / 60.0
        alpha = 0
        if progress < 0.5:
            alpha = int(255 * (progress * 2))
        else:
            alpha = int(255 * (1 - (progress - 0.5) * 2))
        
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((255, 255, 255, alpha))
        self.screen.blit(overlay, (0, 0))

    def _render_game_over(self):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        self.screen.blit(overlay, (0, 0))
        
        msg = "MISSION COMPLETE" if self.win else "MISSION FAILED"
        color = (100, 255, 100) if self.win else (255, 100, 100)
        
        text = self.font_large.render(msg, True, color)
        text_rect = text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
        self.screen.blit(text, text_rect)

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block is for manual play and will not be run by the evaluation.
    # It is included to allow for local testing.
    # Set the video driver to a real one if you want to see the game window.
    if "SDL_VIDEODRIVER" in os.environ and os.environ["SDL_VIDEODRIVER"] == "dummy":
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv()
    obs, info = env.reset()
    
    pygame.display.set_caption("Nebula Harvester")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()

    running = True
    terminated = False
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                terminated = False

        if not terminated:
            keys = pygame.key.get_pressed()
            
            movement = 0 # none
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            space = 1 if keys[pygame.K_SPACE] else 0
            shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [movement, space, shift]
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            if reward != 0:
                print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Terminated: {terminated}")

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Run at 30 FPS

    env.close()