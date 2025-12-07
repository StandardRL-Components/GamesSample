import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:57:11.347964
# Source Brief: brief_00719.md
# Brief Index: 719
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Infiltrate a procedurally generated human cell as a microscopic spy, 
    using word-based abilities to evade defenses and extract vital data.
    """
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Infiltrate a procedurally generated cell as a microscopic spy. Evade biological defenses, "
        "use special abilities, and extract the data core."
    )
    user_guide = (
        "Use arrow keys (↑↓←→) to move. Press space to activate camouflage and shift "
        "for an enzyme boost to destroy nearby enemies."
    )
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 1500

        # Colors (Biopunk theme)
        self.COLOR_BG = (20, 10, 30)
        self.COLOR_WALL = (40, 20, 60)
        self.COLOR_WALL_OUTLINE = (60, 40, 80)
        self.COLOR_PLAYER = (0, 255, 128)
        self.COLOR_PLAYER_GLOW = (0, 255, 128, 50)
        self.COLOR_ENEMY = (255, 50, 50)
        self.COLOR_ENEMY_GLOW = (255, 50, 50, 50)
        self.COLOR_DATA_CORE = (100, 150, 255)
        self.COLOR_DATA_CORE_GLOW = (100, 150, 255, 70)
        self.COLOR_VISION_CONE = (255, 255, 0, 30)
        self.COLOR_VISION_CONE_DETECT = (255, 0, 0, 50)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.COLOR_HEALTH_BAR = (0, 200, 100)
        self.COLOR_HEALTH_BAR_BG = (100, 0, 0)
        self.COLOR_COOLDOWN = (100, 100, 120)

        # Agent properties
        self.AGENT_SPEED = 4.0
        self.AGENT_RADIUS = 10
        self.AGENT_MAX_HEALTH = 100

        # Ability properties
        self.CAMOUFLAGE_DURATION = 90  # 3 seconds at 30fps
        self.CAMOUFLAGE_COOLDOWN = 300 # 10 seconds
        self.ENZYME_BOOST_RADIUS = 50
        self.ENZYME_BOOST_COOLDOWN = 150 # 5 seconds

        # Enemy properties
        self.ENEMY_RADIUS = 8
        self.ENEMY_SPEED = 1.0
        self.ENEMY_VISION_RADIUS = 150
        self.INITIAL_ENEMY_VISION_ANGLE = 30

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_ui = pygame.font.SysFont("Consolas", 18, bold=True)
            self.font_game_over = pygame.font.SysFont("Consolas", 48, bold=True)
        except pygame.error:
            self.font_ui = pygame.font.SysFont(None, 24)
            self.font_game_over = pygame.font.SysFont(None, 60)

        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.agent_pos = pygame.math.Vector2(0, 0)
        self.agent_health = 0
        self.obstacles = []
        self.enemies = []
        self.data_core_pos = pygame.math.Vector2(0, 0)
        self.data_collected = False
        self.particles = []
        
        # Ability states
        self.camouflage_timer = 0
        self.camouflage_cooldown_timer = 0
        self.enzyme_boost_cooldown_timer = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.data_collected = False
        self.particles = []

        self.agent_health = self.AGENT_MAX_HEALTH
        self.camouflage_timer = 0
        self.camouflage_cooldown_timer = 0
        self.enzyme_boost_cooldown_timer = 0

        self._procedural_generation()

        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0

        dist_before = self.agent_pos.distance_to(self.data_core_pos)

        self._handle_input(action)
        self._update_agent()
        detected, detection_penalty = self._update_enemies()
        self._update_particles()
        
        dist_after = self.agent_pos.distance_to(self.data_core_pos)

        # --- Reward Calculation ---
        # Continuous reward for moving towards the goal
        reward += (dist_before - dist_after) * 0.1
        
        # Penalty for being detected
        if detected:
            reward -= 0.5
        
        # --- Event Handling & Rewards ---
        # Enzyme boost kills
        if self._enzyme_boost_triggered:
            killed_enemies = self._perform_enzyme_boost()
            if killed_enemies > 0:
                reward += 5.0 * killed_enemies
                self.score += 10 * killed_enemies
                # sfx: enzyme_boost_hit

        # Agent-enemy collision
        for enemy in self.enemies:
            if self.agent_pos.distance_to(enemy['pos']) < self.AGENT_RADIUS + self.ENEMY_RADIUS:
                self.agent_health -= 20
                reward -= 2.0
                self.agent_pos += (self.agent_pos - enemy['pos']).normalize() * 5 # Knockback
                # sfx: player_damage

        # Data core collection
        if not self.data_collected and self.agent_pos.distance_to(self.data_core_pos) < self.AGENT_RADIUS + 15:
            self.data_collected = True
            self.game_over = True
            reward += 100.0
            self.score += 1000
            # sfx: data_collect_success

        # --- Termination Conditions ---
        terminated = False
        truncated = False
        if self.agent_health <= 0:
            self.agent_health = 0
            self.game_over = True
            terminated = True
            reward = -100.0
            # sfx: game_over_death
        
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            truncated = True
            # No extra penalty, let the negative time-based rewards handle it
        
        if self.data_collected:
            terminated = True

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_input(self, action):
        movement = action[0]
        space_held = action[1] == 1
        shift_held = action[2] == 1

        # Movement
        vel = pygame.math.Vector2(0, 0)
        if movement == 1: vel.y = -1 # Up
        elif movement == 2: vel.y = 1 # Down
        elif movement == 3: vel.x = -1 # Left
        elif movement == 4: vel.x = 1 # Right
        
        if vel.length() > 0:
            vel.normalize_ip()
            self.agent_pos += vel * self.AGENT_SPEED

        # Camouflage (Space)
        if space_held and self.camouflage_cooldown_timer == 0 and self.camouflage_timer == 0:
            self.camouflage_timer = self.CAMOUFLAGE_DURATION
            self.camouflage_cooldown_timer = self.CAMOUFLAGE_COOLDOWN
            self.score -= 5 # Cost to use ability
            # sfx: camouflage_activate

        # Enzyme Boost (Shift)
        self._enzyme_boost_triggered = False
        if shift_held and self.enzyme_boost_cooldown_timer == 0:
            self.enzyme_boost_cooldown_timer = self.ENZYME_BOOST_COOLDOWN
            self._enzyme_boost_triggered = True
            self.score -= 10 # Cost to use ability
            self._create_enzyme_particles()
            # sfx: enzyme_boost_fire

    def _update_agent(self):
        # Update ability timers
        if self.camouflage_timer > 0:
            self.camouflage_timer -= 1
        if self.camouflage_cooldown_timer > 0:
            self.camouflage_cooldown_timer -= 1
        if self.enzyme_boost_cooldown_timer > 0:
            self.enzyme_boost_cooldown_timer -= 1

        # Boundary collision
        self.agent_pos.x = np.clip(self.agent_pos.x, self.AGENT_RADIUS, self.WIDTH - self.AGENT_RADIUS)
        self.agent_pos.y = np.clip(self.agent_pos.y, self.AGENT_RADIUS, self.HEIGHT - self.AGENT_RADIUS)
        
        # Obstacle collision
        for obs in self.obstacles:
            dist = self.agent_pos.distance_to(obs['pos'])
            if dist < self.AGENT_RADIUS + obs['radius']:
                overlap = (self.AGENT_RADIUS + obs['radius']) - dist
                direction = (self.agent_pos - obs['pos']).normalize()
                self.agent_pos += direction * overlap

    def _update_enemies(self):
        is_player_detected = False
        detection_penalty_applied = False

        # Update vision cone angle based on difficulty scaling
        vision_angle = min(90, self.INITIAL_ENEMY_VISION_ANGLE + self.steps // 200)

        for enemy in self.enemies:
            # Patrol behavior
            if enemy['pos'].distance_to(enemy['target']) < 5:
                enemy['target'] = random.choice(enemy['path'])
            
            direction = (enemy['target'] - enemy['pos']).normalize()
            enemy['pos'] += direction * self.ENEMY_SPEED
            enemy['angle'] = math.degrees(math.atan2(-direction.y, direction.x))

            # Detection logic
            is_in_cone = self._is_in_vision_cone(
                self.agent_pos, enemy['pos'], enemy['angle'], 
                vision_angle, self.ENEMY_VISION_RADIUS
            )
            
            if is_in_cone and self.camouflage_timer == 0:
                enemy['detected'] = True
                is_player_detected = True
                if not enemy['was_detected']:
                    detection_penalty_applied = True # Penalize only on first frame of detection
                enemy['was_detected'] = True
            else:
                enemy['detected'] = False
                enemy['was_detected'] = False
        
        return is_player_detected, detection_penalty_applied

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1

    def _perform_enzyme_boost(self):
        killed_count = 0
        enemies_to_remove = []
        for enemy in self.enemies:
            if self.agent_pos.distance_to(enemy['pos']) < self.ENZYME_BOOST_RADIUS:
                enemies_to_remove.append(enemy)
                killed_count += 1
        
        self.enemies = [e for e in self.enemies if e not in enemies_to_remove]
        return killed_count

    def _procedural_generation(self):
        # Place large central nucleus as main obstacle
        self.obstacles = [{
            'pos': pygame.math.Vector2(self.WIDTH / 2 + random.uniform(-50, 50), self.HEIGHT / 2 + random.uniform(-50, 50)),
            'radius': random.uniform(50, 80)
        }]

        # Place smaller organelles
        for _ in range(random.randint(5, 8)):
            while True:
                radius = random.uniform(15, 30)
                pos = pygame.math.Vector2(
                    random.uniform(radius, self.WIDTH - radius),
                    random.uniform(radius, self.HEIGHT - radius)
                )
                # Ensure no overlap with existing obstacles
                if all(pos.distance_to(obs['pos']) > obs['radius'] + radius + 20 for obs in self.obstacles):
                    self.obstacles.append({'pos': pos, 'radius': radius})
                    break
        
        # Find valid spawn locations
        def get_valid_spawn(radius):
            while True:
                pos = pygame.math.Vector2(
                    random.uniform(radius, self.WIDTH - radius),
                    random.uniform(radius, self.HEIGHT - radius)
                )
                if all(pos.distance_to(obs['pos']) > obs['radius'] + radius for obs in self.obstacles):
                    return pos

        self.agent_pos = get_valid_spawn(self.AGENT_RADIUS)
        self.data_core_pos = get_valid_spawn(15)
        # Ensure core is far from player start
        while self.agent_pos.distance_to(self.data_core_pos) < self.WIDTH / 2:
            self.data_core_pos = get_valid_spawn(15)

        # Spawn enemies
        self.enemies = []
        for _ in range(random.randint(3, 5)):
            pos = get_valid_spawn(self.ENEMY_RADIUS)
            path = [get_valid_spawn(self.ENEMY_RADIUS) for _ in range(2)]
            self.enemies.append({
                'pos': pos,
                'path': path,
                'target': random.choice(path),
                'angle': 0,
                'detected': False,
                'was_detected': False
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw obstacles (organelles)
        for obs in self.obstacles:
            pygame.gfxdraw.filled_circle(self.screen, int(obs['pos'].x), int(obs['pos'].y), int(obs['radius']), self.COLOR_WALL)
            pygame.gfxdraw.aacircle(self.screen, int(obs['pos'].x), int(obs['pos'].y), int(obs['radius']), self.COLOR_WALL_OUTLINE)

        # Draw data core
        self._draw_glowing_circle(self.data_core_pos, 15, self.COLOR_DATA_CORE, self.COLOR_DATA_CORE_GLOW, 5)

        # Draw enemies and vision cones
        vision_angle = min(90, self.INITIAL_ENEMY_VISION_ANGLE + self.steps // 200)
        for enemy in self.enemies:
            self._draw_glowing_circle(enemy['pos'], self.ENEMY_RADIUS, self.COLOR_ENEMY, self.COLOR_ENEMY_GLOW, 3)
            self._draw_vision_cone(enemy['pos'], enemy['angle'], vision_angle, self.ENEMY_VISION_RADIUS, enemy['detected'])

        # Draw particles
        for p in self.particles:
            alpha = max(0, 255 * (p['life'] / p['max_life']))
            color = (*p['color'], alpha)
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), int(p['radius']), color)

        # Draw agent
        self._draw_glowing_circle(self.agent_pos, self.AGENT_RADIUS, self.COLOR_PLAYER, self.COLOR_PLAYER_GLOW, 4)
        if self.camouflage_timer > 0:
            self._draw_camouflage_effect()

    def _render_ui(self):
        # Health Bar
        health_ratio = self.agent_health / self.AGENT_MAX_HEALTH
        bar_width = 150
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (10, 10, bar_width, 20))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (10, 10, int(bar_width * health_ratio), 20))
        
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))

        # Cooldown indicators
        ui_y = 40
        def draw_cooldown_bar(label, timer, max_cooldown, y_pos):
            text = self.font_ui.render(label, True, self.COLOR_UI_TEXT)
            self.screen.blit(text, (10, y_pos))
            bar_width = 100
            pygame.draw.rect(self.screen, self.COLOR_COOLDOWN, (10 + text.get_width() + 5, y_pos, bar_width, 15))
            if timer == 0:
                pygame.draw.rect(self.screen, self.COLOR_PLAYER, (10 + text.get_width() + 5, y_pos, bar_width, 15))
            else:
                ratio = 1 - (timer / max_cooldown)
                pygame.draw.rect(self.screen, self.COLOR_DATA_CORE, (10 + text.get_width() + 5, y_pos, int(bar_width * ratio), 15))

        draw_cooldown_bar("CAMO [SPC]", self.camouflage_cooldown_timer, self.CAMOUFLAGE_COOLDOWN, ui_y)
        draw_cooldown_bar("BOOST[SFT]", self.enzyme_boost_cooldown_timer, self.ENZYME_BOOST_COOLDOWN, ui_y + 20)

    def _render_game_over(self):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        
        if self.data_collected:
            msg = "DATA EXTRACTED"
            color = self.COLOR_DATA_CORE
        else:
            msg = "AGENT COMPROMISED"
            color = self.COLOR_ENEMY
            
        text = self.font_game_over.render(msg, True, color)
        text_rect = text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
        self.screen.blit(overlay, (0, 0))
        self.screen.blit(text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.agent_health,
            "data_collected": self.data_collected
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()

    # --- Helper Drawing & Logic Functions ---
    def _draw_glowing_circle(self, pos, radius, color, glow_color, num_layers):
        for i in range(num_layers, 0, -1):
            alpha = glow_color[3] * (1 - (i / num_layers))**2
            current_glow_color = (*glow_color[:3], int(alpha))
            pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), radius + i*2, current_glow_color)
        pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), radius, color)
        pygame.gfxdraw.aacircle(self.screen, int(pos.x), int(pos.y), radius, color)

    def _draw_vision_cone(self, pos, angle, fov, radius, detected):
        color = self.COLOR_VISION_CONE_DETECT if detected else self.COLOR_VISION_CONE
        points = [pos]
        for n in range(-int(fov/2), int(fov/2), 5):
            rad_angle = math.radians(angle + n)
            points.append(pos + pygame.math.Vector2(math.cos(rad_angle), -math.sin(rad_angle)) * radius)
        if len(points) > 2:
            pygame.gfxdraw.aapolygon(self.screen, [(int(p.x), int(p.y)) for p in points], color)
            pygame.gfxdraw.filled_polygon(self.screen, [(int(p.x), int(p.y)) for p in points], color)

    def _is_in_vision_cone(self, point, cone_pos, cone_angle, cone_fov, cone_radius):
        dist = point.distance_to(cone_pos)
        if dist > cone_radius:
            return False
        
        vec_to_point = point - cone_pos
        angle_to_point = math.degrees(math.atan2(-vec_to_point.y, vec_to_point.x))
        
        # Normalize angles to be in [0, 360)
        cone_angle = cone_angle % 360
        angle_to_point = angle_to_point % 360

        # Calculate angular difference
        angle_diff = abs(cone_angle - angle_to_point)
        angle_diff = min(angle_diff, 360 - angle_diff)
        
        return angle_diff <= cone_fov / 2

    def _draw_camouflage_effect(self):
        alpha = 100 * (0.5 + 0.5 * math.sin(self.steps * 0.5))
        color = (*self.COLOR_PLAYER[:3], int(alpha))
        radius = self.AGENT_RADIUS + 5
        pygame.gfxdraw.aacircle(self.screen, int(self.agent_pos.x), int(self.agent_pos.y), radius, color)
        
        # Shimmer particles
        if self.steps % 3 == 0:
            angle = random.uniform(0, 2 * math.pi)
            p_pos = self.agent_pos + pygame.math.Vector2(math.cos(angle), math.sin(angle)) * random.uniform(radius-2, radius+2)
            self.particles.append({'pos': p_pos, 'vel': pygame.math.Vector2(0,0), 'life': 10, 'max_life': 10, 'radius': 2, 'color': self.COLOR_PLAYER})

    def _create_enzyme_particles(self):
        for _ in range(30):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed
            life = random.randint(15, 30)
            self.particles.append({
                'pos': self.agent_pos.copy(),
                'vel': vel,
                'life': life,
                'max_life': life,
                'radius': random.randint(2, 4),
                'color': self.COLOR_PLAYER
            })


if __name__ == '__main__':
    # This block allows you to play the game manually
    # It will not work in a headless environment
    try:
        os.environ["SDL_VIDEODRIVER"] = "x11"
        env = GameEnv(render_mode="rgb_array")
        obs, info = env.reset()
        
        screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        pygame.display.set_caption("Cell Infiltration")
        clock = pygame.time.Clock()
        
        terminated = False
        truncated = False
        total_reward = 0
        
        while not (terminated or truncated):
            # --- Manual Control ---
            keys = pygame.key.get_pressed()
            movement = 0 # No-op
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

            # --- Pygame Event Handling ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True
            
            # --- Rendering ---
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            clock.tick(env.FPS)

            if terminated or truncated:
                print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
                pygame.time.wait(3000)

        env.close()
    except pygame.error as e:
        print(f"Could not run graphical test: {e}")
        print("This is expected in a headless environment.")