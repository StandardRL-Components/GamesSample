import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:03:56.109209
# Source Brief: brief_00787.md
# Brief Index: 787
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

# Helper classes for game objects
class Particle:
    def __init__(self, pos, vel, lifespan, color, size_start, size_end):
        self.pos = list(pos)
        self.vel = list(vel)
        self.lifespan = lifespan
        self.max_lifespan = lifespan
        self.color = color
        self.size_start = size_start
        self.size_end = size_end

    def update(self):
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]
        self.lifespan -= 1

    def draw(self, surface):
        progress = self.lifespan / self.max_lifespan
        current_size = self.size_start * progress + self.size_end * (1 - progress)
        if current_size > 0:
            pygame.draw.circle(surface, self.color, (int(self.pos[0]), int(self.pos[1])), int(current_size))

class Enemy:
    def __init__(self, pos, health, speed, size):
        self.pos = np.array(pos, dtype=float)
        self.health = health
        self.max_health = health
        self.speed = speed
        self.size = size
        self.vel = np.array([0.0, 0.0])

    def update(self, target_pos):
        direction = target_pos - self.pos
        distance = np.linalg.norm(direction)
        if distance > 1:
            self.vel = (direction / distance) * self.speed
        else:
            self.vel = np.array([0.0, 0.0])
        self.pos += self.vel

    def draw(self, surface):
        # Body
        body_color = (200, 20, 20)
        pygame.draw.ellipse(surface, body_color, (*(self.pos - self.size), self.size*2, self.size*2))
        # Eyes
        eye_pos1 = self.pos + self.vel * 2 + np.array([-self.size*0.3, -self.size*0.3])
        eye_pos2 = self.pos + self.vel * 2 + np.array([self.size*0.3, -self.size*0.3])
        pygame.draw.circle(surface, (255, 255, 255), (int(eye_pos1[0]), int(eye_pos1[1])), int(self.size*0.2))
        pygame.draw.circle(surface, (255, 255, 255), (int(eye_pos2[0]), int(eye_pos2[1])), int(self.size*0.2))


class Leaf:
    def __init__(self, pos, vel):
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array(vel, dtype=float)
        self.size = 6
        self.angle = 0
        self.rotation_speed = random.uniform(-5, 5)

    def update(self, gravity):
        self.vel[1] += gravity
        self.pos += self.vel
        self.angle += self.rotation_speed

    def draw(self, surface):
        points = [
            (-self.size, 0), (0, -self.size/2), (self.size, 0), (0, self.size/2)
        ]
        rotated_points = []
        for x, y in points:
            rad = math.radians(self.angle)
            rx = x * math.cos(rad) - y * math.sin(rad)
            ry = x * math.sin(rad) + y * math.cos(rad)
            rotated_points.append((int(self.pos[0] + rx), int(self.pos[1] + ry)))
        
        pygame.gfxdraw.aapolygon(surface, rotated_points, (20, 180, 20))
        pygame.gfxdraw.filled_polygon(surface, rotated_points, (50, 220, 50))


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Defend your giant mushroom from invading creatures by launching leaves from a catapult and building defensive barricades."
    )
    user_guide = (
        "Controls: ←→ to aim, ↑↓ to adjust power. Press space to launch a leaf and shift to build a defensive barricade."
    )
    auto_advance = True
    
    # Constants
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    WIN_CONDITION_STEPS = 1000
    GRAVITY = 0.3

    # Colors
    COLOR_BG = (135, 206, 235)
    COLOR_GROUND = (118, 184, 82)
    COLOR_MUSHROOM_STEM = (222, 184, 135)
    COLOR_MUSHROOM_CAP = (180, 50, 50)
    COLOR_CATAPULT = (92, 64, 51)
    COLOR_LEAF = (50, 220, 50)
    COLOR_ENEMY = (200, 20, 20)
    COLOR_DEFENSE = (139, 69, 19)
    COLOR_UI_TEXT = (255, 255, 255)
    COLOR_UI_BG = (0, 0, 0, 128)

    # Game Parameters
    MUSHROOM_INITIAL_HEALTH = 100
    ANT_INITIAL_POP = 50
    DEFENSE_COST = 10
    LEAF_LAUNCH_COOLDOWN = 10 # steps
    DEFENSE_BUILD_COOLDOWN = 30 # steps

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
        self.font_small = pygame.font.SysFont("Arial", 18, bold=True)
        self.font_large = pygame.font.SysFont("Arial", 48, bold=True)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.mushroom_health = 0
        self.mushroom_pos = np.array([self.WIDTH / 2, self.HEIGHT - 50])
        self.mushroom_size = 0
        self.ant_population = 0
        self.catapult_pos = self.mushroom_pos + np.array([0, -30])
        self.catapult_angle = 0.0
        self.catapult_power = 0.0
        self.enemies = []
        self.leaves = []
        self.defenses = []
        self.particles = []
        self.step_rewards = []
        self.last_leaf_launch_step = -self.LEAF_LAUNCH_COOLDOWN
        self.last_defense_build_step = -self.DEFENSE_BUILD_COOLDOWN
        self.enemy_spawn_timer = 0
        self.enemy_spawn_rate = 50 # steps per spawn
        self.ant_decay_rate = 0.01

        # self.reset() # Removed to avoid double-initialization issues in some frameworks
        # self.validate_implementation() # Removed from init

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.mushroom_health = self.MUSHROOM_INITIAL_HEALTH
        self.mushroom_size = 30
        self.ant_population = self.ANT_INITIAL_POP
        self.catapult_angle = -math.pi / 4
        self.catapult_power = 50
        
        self.enemies.clear()
        self.leaves.clear()
        self.defenses.clear()
        self.particles.clear()
        self.step_rewards.clear()
        
        self.enemy_spawn_timer = 0
        self.enemy_spawn_rate = 50
        self.ant_decay_rate = 0.01

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.step_rewards.clear()
        
        self._handle_input(action)
        self._update_game_logic()
        
        reward = self._calculate_reward()
        terminated = self._check_termination()

        if terminated:
            if self.mushroom_health <= 0:
                reward -= 100 # -100 for mushroom destruction
            elif self.steps >= self.WIN_CONDITION_STEPS:
                reward += 100 # +100 for survival
            self.game_over = True
        
        self.steps += 1
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Movement: adjust catapult aim
        if movement == 1: self.catapult_power = min(100, self.catapult_power + 2) # Up
        if movement == 2: self.catapult_power = max(20, self.catapult_power - 2) # Down
        if movement == 3: self.catapult_angle -= 0.05 # Left
        if movement == 4: self.catapult_angle += 0.05 # Right
        self.catapult_angle = np.clip(self.catapult_angle, -math.pi * 0.9, -math.pi * 0.1)

        # Action 1: Launch leaf
        if space_held and (self.steps - self.last_leaf_launch_step >= self.LEAF_LAUNCH_COOLDOWN):
            self._launch_leaf()
            self.last_leaf_launch_step = self.steps
            # sfx: catapult_launch.wav

        # Action 2: Build defense
        if shift_held and (self.steps - self.last_defense_build_step >= self.DEFENSE_BUILD_COOLDOWN):
            if self.ant_population >= self.DEFENSE_COST:
                self._build_defense()
                self.last_defense_build_step = self.steps
                # sfx: build_defense.wav

    def _launch_leaf(self):
        power_scaled = self.catapult_power / 10.0
        vel = np.array([
            math.cos(self.catapult_angle) * power_scaled,
            math.sin(self.catapult_angle) * power_scaled
        ])
        launch_pos = self.catapult_pos + np.array([math.cos(self.catapult_angle), math.sin(self.catapult_angle)]) * 30
        self.leaves.append(Leaf(launch_pos, vel))

    def _build_defense(self):
        self.ant_population -= self.DEFENSE_COST
        
        angle = random.uniform(0, 2 * math.pi)
        radius = self.mushroom_size + 40 + random.uniform(0, 20)
        pos = self.mushroom_pos + np.array([math.cos(angle) * radius, math.sin(angle) * radius - 30])
        pos[1] = min(pos[1], self.HEIGHT - 50 - 10) # Ensure it's on the ground
        
        # Simple non-overlapping check
        can_place = True
        for d in self.defenses:
            if np.linalg.norm(pos - d['pos']) < 30:
                can_place = False
                break
        
        if can_place:
            self.defenses.append({'pos': pos, 'size': 15, 'health': 100})
            self.step_rewards.append(-0.1 * self.DEFENSE_COST) # Reward ant usage
        else:
            self.ant_population += self.DEFENSE_COST # Refund if failed

    def _update_game_logic(self):
        # Update difficulty
        if self.steps > 0 and self.steps % 200 == 0:
            self.enemy_spawn_rate = max(10, self.enemy_spawn_rate * 0.9)
        if self.steps > 0 and self.steps % 400 == 0:
            self.ant_decay_rate *= 1.5

        # Update ants
        ants_lost = self.ant_decay_rate / self.FPS
        self.ant_population = max(0, self.ant_population - ants_lost)
        if ants_lost > 0: self.step_rewards.append(-0.1)

        # Update mushroom (grows slowly)
        self.mushroom_size += 0.01
        
        # Spawn enemies
        self.enemy_spawn_timer += 1
        if self.enemy_spawn_timer >= self.enemy_spawn_rate:
            self.enemy_spawn_timer = 0
            side = random.choice([-1, 1])
            pos = [self.WIDTH if side > 0 else 0, self.HEIGHT - 60]
            self.enemies.append(Enemy(pos, health=30, speed=random.uniform(0.8, 1.5), size=12))

        # Update entities
        for leaf in self.leaves[:]:
            leaf.update(self.GRAVITY)
            if not (0 < leaf.pos[0] < self.WIDTH and leaf.pos[1] < self.HEIGHT):
                self.leaves.remove(leaf)

        for enemy in self.enemies:
            # Basic pathfinding around defenses
            target = self.mushroom_pos
            for defense in self.defenses:
                if np.linalg.norm(enemy.pos - defense['pos']) < 40:
                    # Nudge away from defense
                    nudge_dir = enemy.pos - defense['pos']
                    target += nudge_dir * 0.5
            enemy.update(target)

        for particle in self.particles[:]:
            particle.update()
            if particle.lifespan <= 0:
                self.particles.remove(particle)

        # Check collisions
        self._check_collisions()

    def _check_collisions(self):
        # Leaves vs Enemies
        for leaf in self.leaves[:]:
            for enemy in self.enemies[:]:
                if np.linalg.norm(leaf.pos - enemy.pos) < enemy.size + leaf.size:
                    self.leaves.remove(leaf)
                    enemy.health -= 20
                    self.step_rewards.append(0.1) # Reward for hit
                    self._create_impact_particles(enemy.pos, (255, 255, 0), 5)
                    # sfx: hit_enemy.wav
                    if enemy.health <= 0:
                        self.enemies.remove(enemy)
                        self.score += 1
                        self.step_rewards.append(1.0) # Reward for kill
                        self._create_impact_particles(enemy.pos, self.COLOR_ENEMY, 15)
                        # sfx: enemy_die.wav
                    break
        
        # Enemies vs Mushroom
        for enemy in self.enemies[:]:
            if np.linalg.norm(enemy.pos - self.mushroom_pos) < self.mushroom_size + enemy.size:
                self.enemies.remove(enemy)
                self.mushroom_health -= 10
                self.score -= 2 # Penalty for letting enemy through
                self._create_impact_particles(self.mushroom_pos, self.COLOR_MUSHROOM_CAP, 10)
                # sfx: mushroom_damage.wav

        # Enemies vs Defenses
        for enemy in self.enemies:
            for defense in self.defenses:
                if np.linalg.norm(enemy.pos - defense['pos']) < defense['size'] + enemy.size:
                    enemy.pos -= enemy.vel * 1.5 # Push back

    def _create_impact_particles(self, pos, color, count):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = random.randint(15, 30)
            self.particles.append(Particle(pos, vel, lifespan, color, 5, 0))

    def _calculate_reward(self):
        return sum(self.step_rewards)

    def _check_termination(self):
        return self.mushroom_health <= 0 or self.steps >= self.WIN_CONDITION_STEPS

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "mushroom_health": self.mushroom_health,
            "ant_population": self.ant_population,
        }

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        if self.game_over:
            self._render_game_over()
            
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Ground
        pygame.draw.rect(self.screen, self.COLOR_GROUND, (0, self.HEIGHT - 50, self.WIDTH, 50))
        
        # Defenses
        for d in self.defenses:
            pygame.draw.circle(self.screen, self.COLOR_DEFENSE, (int(d['pos'][0]), int(d['pos'][1])), d['size'])

        # Mushroom
        stem_rect = pygame.Rect(0, 0, self.mushroom_size * 0.6, self.mushroom_size)
        stem_rect.center = (self.mushroom_pos[0], self.mushroom_pos[1] + self.mushroom_size * 0.3)
        pygame.draw.rect(self.screen, self.COLOR_MUSHROOM_STEM, stem_rect, border_radius=5)
        pygame.gfxdraw.filled_circle(self.screen, int(self.mushroom_pos[0]), int(self.mushroom_pos[1]), int(self.mushroom_size), self.COLOR_MUSHROOM_CAP)
        pygame.gfxdraw.aacircle(self.screen, int(self.mushroom_pos[0]), int(self.mushroom_pos[1]), int(self.mushroom_size), (0,0,0,50))

        # Catapult
        arm_length = 35
        arm_end_x = self.catapult_pos[0] + arm_length * math.cos(self.catapult_angle)
        arm_end_y = self.catapult_pos[1] + arm_length * math.sin(self.catapult_angle)
        pygame.draw.line(self.screen, self.COLOR_CATAPULT, (int(self.catapult_pos[0]), int(self.catapult_pos[1])), (int(arm_end_x), int(arm_end_y)), 8)
        pygame.draw.circle(self.screen, self.COLOR_CATAPULT, (int(self.catapult_pos[0]), int(self.catapult_pos[1])), 10)

        # Trajectory guide
        for i in range(1, 6):
            t = i * 0.15
            power_scaled = self.catapult_power / 10.0
            dx = self.catapult_pos[0] + (power_scaled * math.cos(self.catapult_angle) * t * 10)
            dy = self.catapult_pos[1] + (power_scaled * math.sin(self.catapult_angle) * t * 10) + (0.5 * self.GRAVITY * (t*10)**2)
            pygame.draw.circle(self.screen, (255, 255, 255, 100), (int(dx), int(dy)), 2)

        # Entities
        for leaf in self.leaves:
            leaf.draw(self.screen)
        for enemy in self.enemies:
            enemy.draw(self.screen)
        for particle in self.particles:
            particle.draw(self.screen)

    def _render_ui(self):
        # UI Panel
        ui_panel = pygame.Surface((self.WIDTH, 40), pygame.SRCALPHA)
        ui_panel.fill(self.COLOR_UI_BG)
        self.screen.blit(ui_panel, (0, 0))

        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Ant Population
        ant_text = self.font_small.render(f"ANTS: {int(self.ant_population)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(ant_text, (150, 10))

        # Mushroom Health
        health_text = self.font_small.render(f"HOME: {int(self.mushroom_health)}%", True, self.COLOR_UI_TEXT)
        self.screen.blit(health_text, (280, 10))
        
        # Timer
        time_left = max(0, self.WIN_CONDITION_STEPS - self.steps)
        timer_text = self.font_small.render(f"TIME: {time_left}", True, self.COLOR_UI_TEXT)
        self.screen.blit(timer_text, (self.WIDTH - 100, 10))
        
        # Power Meter
        power_bar_bg = pygame.Rect(self.WIDTH/2 - 50, 10, 100, 20)
        pygame.draw.rect(self.screen, (80, 80, 80), power_bar_bg, border_radius=3)
        power_width = int((self.catapult_power / 100) * 96)
        power_bar_fill = pygame.Rect(self.WIDTH/2 - 48, 12, power_width, 16)
        pygame.draw.rect(self.screen, (255, 200, 0), power_bar_fill, border_radius=3)


    def _render_game_over(self):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        self.screen.blit(overlay, (0, 0))

        if self.mushroom_health <= 0:
            msg = "DEFEAT"
            color = self.COLOR_ENEMY
        else:
            msg = "VICTORY!"
            color = (255, 215, 0)
        
        text = self.font_large.render(msg, True, color)
        text_rect = text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
        self.screen.blit(text, text_rect)

    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        # print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually
    # For this to work, you must comment out the line:
    # os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    
    # Re-enable the display for manual play
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    running = True
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Mushroom Defense")
    clock = pygame.time.Clock()

    action = np.array([0, 0, 0]) # [movement, space, shift]

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        keys = pygame.key.get_pressed()
        
        # Map keyboard to MultiDiscrete action space
        mov = 0 # None
        if keys[pygame.K_UP] or keys[pygame.K_w]: mov = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: mov = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: mov = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: mov = 4
        
        space = 1 if keys[pygame.K_SPACE] else 0
        shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = np.array([mov, space, shift])

        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
            # Wait for a moment before reset
            pygame.time.wait(2000)
            obs, info = env.reset()

        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.FPS)

    env.close()